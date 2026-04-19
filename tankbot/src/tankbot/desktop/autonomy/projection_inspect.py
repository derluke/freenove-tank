"""Diagnostic renderer for the depth -> robot-frame -> 2D-scan pipeline.

Loads a single frame from a recording, runs monocular depth, projects to
robot-frame points, and writes a PNG with four panels:

  1. RGB input image (reference).
  2. Depth map, colormapped.
  3. Per-pixel `z_rob` classified as below-floor / scan-band / above-scan.
     This is the key diagnostic: every pixel painted "scan-band" will
     contribute to the 2D scan as an obstacle. If a large part of the
     lower image (floor) lands in that band, extrinsics or depth scale
     are off.
  4. Top-down view of the projected cloud, with scan-band points drawn
     in color and below/above-band points in gray for context.

Usage:
    task projection:inspect DATASET=recordings/phase0c/straight_1m_1

The default frame is the middle of the recording; pass FRAME=<index> to
pick a specific one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tankbot.desktop.autonomy.dataset import iter_frame_records
from tankbot.desktop.autonomy.depth import build_estimator
from tankbot.desktop.autonomy.reactive import load_extrinsics, load_intrinsics
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
    _effective_extrinsics,
    _normalized_camera_xy,
)
from tankbot.desktop.autonomy.reactive.scan import ScanConfig

_DEFAULT_BACKEND = "depth_anything_v2"
_DEFAULT_INTRINSICS = "calibration/camera_intrinsics.yaml"
_DEFAULT_EXTRINSICS = "calibration/camera_extrinsics.yaml"
_DEFAULT_DEPTH_SCALE = 1.0

# Panel size: each panel is rendered at this resolution and then stacked.
_PANEL_W = 640
_PANEL_H = 480


@dataclass(frozen=True)
class PerPixelProjection:
    """Per-pixel diagnostic arrays, all shape HxW.

    Pixels that fail range/validity checks are NaN in `z_rob` / `x_rob` /
    `y_rob`, and `classification` is 0 (background).
    """

    z_rob: np.ndarray       # robot-frame height (m), NaN where invalid
    x_rob: np.ndarray       # robot-frame forward (m), NaN where invalid
    y_rob: np.ndarray       # robot-frame left (m), NaN where invalid
    range_m: np.ndarray     # 2D range from robot origin (m), NaN where invalid
    classification: np.ndarray  # 0=invalid, 1=below, 2=scan-band, 3=above


def project_per_pixel(
    depth_m: np.ndarray,
    intrinsics,
    extrinsics,
    proj_cfg: ProjectionConfig,
    scan_cfg: ScanConfig,
) -> PerPixelProjection:
    """Run the full projection pipeline at native resolution, *per pixel*.

    This deliberately mirrors `project_depth_to_robot_points` but keeps
    the HxW layout so we can render which-pixels-become-obstacles. It
    bypasses the downsample step since the point here is visualization.
    """
    if depth_m.ndim != 2:
        msg = f"depth_m must be HxW, got {depth_m.shape}"
        raise ValueError(msg)
    h, w = depth_m.shape
    depth = depth_m.astype(np.float32)
    if proj_cfg.depth_scale != 1.0:
        depth = depth * proj_cfg.depth_scale

    finite = np.isfinite(depth)
    in_range = (depth >= proj_cfg.min_range_m) & (depth <= proj_cfg.max_range_m)
    valid = finite & in_range

    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    x_rob = np.full((h, w), np.nan, dtype=np.float32)
    y_rob = np.full((h, w), np.nan, dtype=np.float32)
    z_rob = np.full((h, w), np.nan, dtype=np.float32)

    if valid.any():
        z = depth[valid].astype(np.float32)
        u = xs[valid]
        v = ys[valid]
        x_norm, y_norm = _normalized_camera_xy(u, v, intrinsics, proj_cfg)
        p_cam = np.stack([x_norm * z, y_norm * z, z], axis=-1)

        effective_extrinsics = _effective_extrinsics(extrinsics, proj_cfg)
        rotation = effective_extrinsics.rotation_robot_from_camera().astype(np.float32)
        translation = effective_extrinsics.translation_robot_from_camera().astype(np.float32)
        p_rob = p_cam @ rotation.T + translation

        x_rob[valid] = p_rob[:, 0]
        y_rob[valid] = p_rob[:, 1]
        z_rob[valid] = p_rob[:, 2]

    range_m = np.sqrt(x_rob * x_rob + y_rob * y_rob)

    classification = np.zeros((h, w), dtype=np.uint8)
    below = valid & (z_rob < scan_cfg.min_height_m)
    band = valid & (z_rob >= scan_cfg.min_height_m) & (z_rob <= scan_cfg.max_height_m)
    above = valid & (z_rob > scan_cfg.max_height_m)
    classification[below] = 1
    classification[band] = 2
    classification[above] = 3

    return PerPixelProjection(
        z_rob=z_rob,
        x_rob=x_rob,
        y_rob=y_rob,
        range_m=range_m,
        classification=classification,
    )


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        out, text, (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def render_rgb_panel(bgr: np.ndarray) -> np.ndarray:
    return _label(_fit(bgr, _PANEL_W, _PANEL_H), "RGB")


def render_depth_panel(depth_m: np.ndarray, scale: float) -> np.ndarray:
    scaled = depth_m.astype(np.float32) * scale
    finite = np.isfinite(scaled)
    if not finite.any():
        panel = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
        return _label(panel, "depth (no valid pixels)")
    vmin = float(np.nanmin(np.where(finite, scaled, np.nan)))
    vmax = float(np.nanpercentile(scaled[finite], 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    norm = np.clip((scaled - vmin) / (vmax - vmin), 0.0, 1.0)
    norm[~finite] = 0.0
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[~finite] = (40, 40, 40)
    return _label(
        _fit(colored, _PANEL_W, _PANEL_H),
        f"depth (after scale={scale:.2f}) range=[{vmin:.2f}, {vmax:.2f}] m",
    )


def render_classification_panel(
    bgr: np.ndarray,
    proj: PerPixelProjection,
    scan_cfg: ScanConfig,
) -> np.ndarray:
    """Overlay per-pixel scan-band classification on the RGB image.

    - invalid pixels: show raw image at 40% brightness
    - below scan band (floor): teal tint
    - inside scan band (obstacle): red tint
    - above scan band (ceiling/upper): blue tint
    """
    base = (bgr.astype(np.float32) * 0.4).astype(np.uint8)
    out = base.copy()
    cls = proj.classification

    floor_mask = cls == 1
    band_mask = cls == 2
    above_mask = cls == 3

    tint = out.astype(np.float32)
    tint[floor_mask] = tint[floor_mask] * 0.5 + np.array([120, 180, 60], dtype=np.float32) * 0.5
    tint[band_mask]  = tint[band_mask]  * 0.4 + np.array([60, 60, 240], dtype=np.float32) * 0.6
    tint[above_mask] = tint[above_mask] * 0.7 + np.array([220, 120, 60], dtype=np.float32) * 0.3
    out = np.clip(tint, 0, 255).astype(np.uint8)

    total = cls.size
    f_pct = 100.0 * floor_mask.sum() / total
    b_pct = 100.0 * band_mask.sum() / total
    a_pct = 100.0 * above_mask.sum() / total
    label = (
        f"z_rob bands | floor<{scan_cfg.min_height_m:.2f}m:{f_pct:.0f}% "
        f"scan[{scan_cfg.min_height_m:.2f},{scan_cfg.max_height_m:.2f}]:{b_pct:.0f}% "
        f"above>{scan_cfg.max_height_m:.2f}:{a_pct:.0f}%"
    )
    return _label(_fit(out, _PANEL_W, _PANEL_H), label)


def render_topdown_panel(
    proj: PerPixelProjection,
    scan_cfg: ScanConfig,
    x_range_m: float = 5.0,
    y_range_m: float = 3.0,
) -> np.ndarray:
    """Top-down plot: x forward (up), y left (left).

    Scan-band pixels drawn red; floor (below) drawn faint teal; above-band
    drawn faint blue. Robot origin marked, scan-band envelope shown.
    """
    panel = np.full((_PANEL_H, _PANEL_W, 3), 255, dtype=np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        # x forward -> px goes up; y left -> px goes left.
        col = int(_PANEL_W / 2 - y / y_range_m * (_PANEL_W / 2))
        row = int(_PANEL_H - 10 - x / x_range_m * (_PANEL_H - 20))
        return col, row

    # Grid lines at 1 m.
    for x_m in np.arange(0, x_range_m + 0.01, 1.0):
        _, row = to_px(float(x_m), 0.0)
        cv2.line(panel, (0, row), (_PANEL_W, row), (220, 220, 220), 1)
        cv2.putText(panel, f"{x_m:.0f}m", (4, row - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    for y_m in np.arange(-y_range_m, y_range_m + 0.01, 1.0):
        col, _ = to_px(0.0, float(y_m))
        cv2.line(panel, (col, 0), (col, _PANEL_H), (220, 220, 220), 1)

    cls = proj.classification
    x = proj.x_rob
    y = proj.y_rob

    def draw_class(mask: np.ndarray, color: tuple[int, int, int], size: int) -> None:
        xs = x[mask]
        ys = y[mask]
        # Subsample to keep render fast.
        if xs.size > 8000:
            idx = np.random.default_rng(0).choice(xs.size, 8000, replace=False)
            xs = xs[idx]
            ys = ys[idx]
        for xv, yv in zip(xs.tolist(), ys.tolist(), strict=True):
            col, row = to_px(xv, yv)
            if 0 <= col < _PANEL_W and 0 <= row < _PANEL_H:
                cv2.circle(panel, (col, row), size, color, -1)

    draw_class(cls == 1, (200, 220, 180), 1)  # floor faint teal
    draw_class(cls == 3, (220, 200, 180), 1)  # above faint blue
    draw_class(cls == 2, (40, 40, 220), 2)    # scan-band red

    # Robot marker at origin.
    ocol, orow = to_px(0.0, 0.0)
    cv2.circle(panel, (ocol, orow), 8, (0, 0, 0), 2)
    cv2.line(panel, (ocol, orow), (ocol, orow - 16), (0, 0, 0), 2)

    return _label(
        panel,
        f"top-down (red=scan-band, z in [{scan_cfg.min_height_m:.2f},{scan_cfg.max_height_m:.2f}] m)",
    )


def compose_panels(panels: list[np.ndarray]) -> np.ndarray:
    # 2x2 grid.
    row1 = np.hstack([panels[0], panels[1]])
    row2 = np.hstack([panels[2], panels[3]])
    return np.vstack([row1, row2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int, default=-1,
        help="Frame index to inspect. -1 (default) = middle frame.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path. Defaults to <dataset>/inspect_f<idx>.png.")
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    parser.add_argument("--intrinsics", type=Path, default=Path(_DEFAULT_INTRINSICS))
    parser.add_argument("--extrinsics", type=Path, default=Path(_DEFAULT_EXTRINSICS))
    parser.add_argument("--depth-scale", type=float, default=_DEFAULT_DEPTH_SCALE)
    args = parser.parse_args(argv)

    frames = list(iter_frame_records(args.dataset))
    if not frames:
        print(f"No frames found in {args.dataset}", file=sys.stderr)
        return 2
    idx = args.frame if args.frame >= 0 else len(frames) // 2
    if idx >= len(frames):
        print(f"frame {idx} out of range (dataset has {len(frames)})", file=sys.stderr)
        return 2
    frame = frames[idx]

    bgr = cv2.imread(str(frame.path))
    if bgr is None:
        print(f"cannot read {frame.path}", file=sys.stderr)
        return 2

    intrinsics = load_intrinsics(args.intrinsics)
    extrinsics = load_extrinsics(args.extrinsics)

    print(f"Loading depth backend: {args.backend} ...")
    estimator = build_estimator(args.backend)
    estimator.load()
    print(f"Inferring frame {idx} ({frame.path.name}) ...")
    depth_result = estimator.infer(bgr)

    proj_cfg = ProjectionConfig(depth_scale=args.depth_scale)
    scan_cfg = ScanConfig()
    proj = project_per_pixel(
        depth_result.depth_m, intrinsics, extrinsics, proj_cfg, scan_cfg,
    )

    panels = [
        render_rgb_panel(bgr),
        render_depth_panel(depth_result.depth_m, args.depth_scale),
        render_classification_panel(bgr, proj, scan_cfg),
        render_topdown_panel(proj, scan_cfg),
    ]
    composed = compose_panels(panels)

    # Header with extrinsics for quick reference.
    header_h = 30
    header = np.full((header_h, composed.shape[1], 3), 20, dtype=np.uint8)
    hdr = (
        f"{args.dataset.name} frame {idx}/{len(frames)-1}  "
        f"extrinsics: h={extrinsics.height_m:.2f}m pitch={extrinsics.pitch_deg:.1f}deg  "
        f"depth_scale={args.depth_scale:.2f}"
    )
    cv2.putText(header, hdr, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    composed = np.vstack([header, composed])

    out_path = args.output or (args.dataset / f"inspect_f{idx:05d}.png")
    cv2.imwrite(str(out_path), composed)
    print(f"Wrote {out_path}")

    # Print floor-band stats to the console for quick diagnosis.
    total = proj.classification.size
    h, _w = proj.classification.shape
    lower_half = proj.classification[h // 2:, :]
    lower_band_pct = 100.0 * (lower_half == 2).sum() / lower_half.size
    print(
        "Per-pixel z_rob bands (whole image) | "
        f"floor={100.0*(proj.classification==1).sum()/total:.1f}%  "
        f"scan-band={100.0*(proj.classification==2).sum()/total:.1f}%  "
        f"above={100.0*(proj.classification==3).sum()/total:.1f}%  "
        f"invalid={100.0*(proj.classification==0).sum()/total:.1f}%"
    )
    print(
        f"Lower-half (expected mostly floor): {lower_band_pct:.1f}% of pixels "
        "fall in the scan band — ideal value is ~0%."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
