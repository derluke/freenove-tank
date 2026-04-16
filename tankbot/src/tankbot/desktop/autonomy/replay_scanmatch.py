"""Replay a recorded dataset through the Phase 2 scan-match pipeline.

For each frame:
1. Load the BGR image.
2. Run Depth Anything V2 inference to get a metric depth map.
3. Project depth to robot-frame 3D points.
4. Feed points to ScanMatchPoseSource to get a frame-to-frame ICP
   match and accumulated world-frame pose.
5. Plot obstacle points in world frame to build a 2D map.
6. Render a visualization panel: camera | depth | accumulated map
   with pose trail | match stats.

Usage:
    tankbot-replay-scanmatch --dataset recordings/mast3r_raw/vanilla1
    tankbot-replay-scanmatch --dataset recordings/mast3r_raw/vanilla1 --save-dir logs/scanmatch_replay
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tankbot.desktop.autonomy.dataset import iter_frame_records
from tankbot.desktop.autonomy.depth import build_estimator
from tankbot.desktop.autonomy.reactive import load_extrinsics, load_intrinsics
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
    project_depth_to_robot_points,
)
from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)
from tankbot.desktop.autonomy.reactive.world_map import WorldMap

_DEFAULT_DEPTH_BACKEND = "depth_anything_v2"
_DEFAULT_EXTRINSICS = "calibration/camera_extrinsics.yaml"
_DEFAULT_INTRINSICS = "calibration/camera_intrinsics.yaml"

# Visualization layout.
_PANEL_H = 240
_PANEL_W = 320


def _colorize_depth(depth_m: np.ndarray, max_m: float = 5.0) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, max_m)
    norm = (clipped / max_m * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _render_stats(
    pose_source: ScanMatchPoseSource,
    depth_ms: float,
    match_ms: float,
    frame_idx: int,
    n_points: int,
) -> np.ndarray:
    """Render a text panel with scan-match stats."""
    h = _PANEL_H
    w = _PANEL_W
    img = np.zeros((h, w, 3), dtype=np.uint8)

    pose = pose_source.latest()
    match = pose_source.last_match

    lines = [
        f"Frame {frame_idx}",
        f"Health: {pose.health.value}",
        f"Source: {pose.source}",
        "",
        f"Pose: ({pose.xy_m[0]:.3f}, {pose.xy_m[1]:.3f})" if pose.xy_m else "Pose: None",
        f"Yaw:  {math.degrees(pose.yaw_rad):.1f} deg",
        "",
    ]

    if match is not None:
        lines.extend(
            [
                f"ICP: {'converged' if match.converged else 'FAILED'}",
                f"  dx={match.dx:.4f} dy={match.dy:.4f}",
                f"  dyaw={math.degrees(match.dyaw):.2f} deg",
                f"  residual={match.mean_residual:.4f} m",
                f"  corr={match.n_correspondences}  iter={match.iterations}",
            ]
        )
    else:
        lines.append("ICP: first frame")

    lines.extend(
        [
            "",
            f"Depth:  {depth_ms:.1f} ms",
            f"Match:  {match_ms:.1f} ms",
            f"Points: {n_points}",
        ]
    )

    y = 14
    for line in lines:
        cv2.putText(img, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 14

    return img


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay dataset through Phase 2 scan-match pipeline.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to recording directory.")
    parser.add_argument("--backend", default=_DEFAULT_DEPTH_BACKEND, help="Depth backend name.")
    parser.add_argument("--extrinsics", type=Path, default=Path(_DEFAULT_EXTRINSICS))
    parser.add_argument("--intrinsics", type=Path, default=Path(_DEFAULT_INTRINSICS))
    parser.add_argument("--save-dir", type=Path, default=None, help="Save annotated frames here.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--headless", action="store_true", help="No imshow window.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth scale correction (ultrasonic_m / depth_median).",
    )
    args = parser.parse_args(argv)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration.
    intrinsics = load_intrinsics(args.intrinsics)
    extrinsics = load_extrinsics(args.extrinsics)
    proj_cfg = ProjectionConfig(depth_scale=args.depth_scale)

    # Load depth backend.
    print(f"Loading depth backend: {args.backend} ...")
    estimator = build_estimator(args.backend)
    estimator.load()
    print("Depth backend ready.")

    # Build scan-match pose source.
    scan_cfg = ScanConfig()
    icp_cfg = ICPConfig()
    sm_cfg = ScanMatchConfig()
    pose_source = ScanMatchPoseSource(
        scan_config=scan_cfg,
        icp_config=icp_cfg,
        config=sm_cfg,
    )

    # World map accumulator.
    world_map = WorldMap()

    # Iterate frames.
    frames = list(iter_frame_records(args.dataset))
    if not frames:
        print(f"No frames found in {args.dataset}", file=sys.stderr)
        sys.exit(1)

    if args.max_frames:
        frames = frames[: args.max_frames]

    print(f"Replaying {len(frames)} frames ...")

    for i, frame_rec in enumerate(frames):
        bgr = cv2.imread(str(frame_rec.path))
        if bgr is None:
            print(f"  skip {frame_rec.path} (unreadable)")
            continue

        # Depth inference.
        t0 = time.monotonic()
        depth_result = estimator.infer(bgr)
        depth_ms = (time.monotonic() - t0) * 1000.0

        # Project to robot-frame 3D points.
        points = project_depth_to_robot_points(
            depth_result.depth_m,
            intrinsics,
            extrinsics,
            proj_cfg,
        )

        # Scan match.
        t_match = time.monotonic()
        pose_source.update(points)
        match_ms = (time.monotonic() - t_match) * 1000.0

        # Get current pose.
        pose = pose_source.latest()
        rx, ry = pose.xy_m if pose.xy_m is not None else (0.0, 0.0)
        ryaw = pose.yaw_rad

        # Accumulate into world map (only when pose is healthy).
        from tankbot.desktop.autonomy.reactive.pose import HealthState

        good = pose.health == HealthState.HEALTHY and pose.xy_m is not None
        world_map.add_frame(rx, ry, ryaw, points, good_pose=good)

        # Visualization.
        depth_color = _colorize_depth(depth_result.depth_m)
        map_img = world_map.render()
        stats_img = _render_stats(pose_source, depth_ms, match_ms, frame_rec.index, points.shape[0])

        # Compose 2x2 panel.
        top_left = cv2.resize(bgr, (_PANEL_W, _PANEL_H))
        top_right = cv2.resize(depth_color, (_PANEL_W, _PANEL_H))
        top = np.hstack([top_left, top_right])

        map_resized = cv2.resize(map_img, (_PANEL_W, _PANEL_H))
        bottom = np.hstack([map_resized, stats_img])
        panel = np.vstack([top, bottom])

        if args.save_dir:
            out_path = args.save_dir / f"scanmatch_{frame_rec.index:06d}.png"
            cv2.imwrite(str(out_path), panel)

        if not args.headless:
            cv2.imshow("Scan Match Replay", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 in (ord(" "), ord("q")):
                        break
                if key2 == ord("q"):
                    break

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            match = pose_source.last_match
            res_str = f"res={match.mean_residual:.4f}" if match else "first"
            print(
                f"  [{i + 1}/{len(frames)}] depth={depth_ms:.0f}ms "
                f"match={match_ms:.1f}ms "
                f"pose=({rx:.3f},{ry:.3f}) yaw={math.degrees(ryaw):.1f}deg "
                f"{pose.health.value} {res_str}"
            )

    if not args.headless:
        cv2.destroyAllWindows()

    estimator.unload()
    print("Done.")


if __name__ == "__main__":
    main()
