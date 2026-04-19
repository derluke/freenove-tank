"""Replay a recorded dataset through the Phase 1 reactive safety stack.

For each frame event in monotonic time order:
1. Load the BGR image.
2. Run Depth Anything V2 inference to get a metric depth map.
3. Feed the depth map plus the latest IMU/telemetry state through
   `ReactiveSafety.tick()`.
4. Render an annotated visualization panel: camera frame | colorized
   depth | occupancy grid (top-down) | text overlay with motor intent,
   veto state, clearances, and timing.
5. Display with OpenCV `imshow` (press 'q' to quit, space to pause).
   Optionally save frames to an output directory.

Usage:
    tankbot-replay-reactive --dataset recordings/mast3r_raw/vanilla1
    tankbot-replay-reactive --dataset recordings/mast3r_raw/vanilla1 --save-dir logs/reactive_replay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tankbot.desktop.autonomy.dataset import (
    FrameRecord,
    ImuRecord,
    TelemetryRecord,
    iter_dataset_events,
)
from tankbot.desktop.autonomy.depth import build_estimator
from tankbot.desktop.autonomy.reactive import (
    GyroPoseSource,
    GyroIntegrator,
    ReactiveSafety,
    SafetyTick,
    load_extrinsics,
    load_intrinsics,
)
from tankbot.desktop.autonomy.reactive.grid import (
    CELL_OCCUPIED,
    ReactiveGrid,
)

# Defaults — override via CLI flags.
_DEFAULT_DEPTH_BACKEND = "depth_anything_v2"
_DEFAULT_EXTRINSICS = "calibration/camera_extrinsics.yaml"
_DEFAULT_INTRINSICS = "calibration/camera_intrinsics.yaml"

# Visualization layout.
_PANEL_H = 240
_PANEL_W = 320
_GRID_PANEL_SIZE = 240  # px, square


def _colorize_depth(depth_m: np.ndarray, max_m: float = 5.0) -> np.ndarray:
    """Depth map -> BGR colorized image for visualization."""
    clipped = np.clip(depth_m, 0.0, max_m)
    norm = (clipped / max_m * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _render_grid(grid: ReactiveGrid, tick: SafetyTick) -> np.ndarray:
    """Render the occupancy grid as a top-down BGR image.

    Robot is at center, +x (forward) points up. Occupied cells are red,
    unknown cells are dark gray. A green dot marks the robot origin.
    """
    cells = grid.cells
    size = grid.size

    # Build an RGB image.
    img = np.full((size, size, 3), 40, dtype=np.uint8)  # dark gray
    occupied_mask = cells == CELL_OCCUPIED
    img[occupied_mask] = (0, 0, 220)  # red in BGR

    # Flip so forward (+x = higher row index) is up in the image.
    # np.flip + contiguous copy so OpenCV can draw on it.
    img = np.ascontiguousarray(img[::-1, :, :])

    # Mark robot center (green dot).
    center = size // 2
    cv2.circle(img, (center, center), 2, (0, 220, 0), -1)

    # Stop zone indicator.
    if tick.stop_zone_blocked:
        cv2.putText(img, "BLOCKED", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Scale up to the target panel size.
    return cv2.resize(img, (_GRID_PANEL_SIZE, _GRID_PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


def _render_text_panel(tick: SafetyTick, depth_ms: float, frame_idx: int) -> np.ndarray:
    """Render a text overlay panel with decision details."""
    h = _GRID_PANEL_SIZE
    w = _PANEL_W
    img = np.zeros((h, w, 3), dtype=np.uint8)

    lines = [
        f"Frame {frame_idx}",
        f"Decision: {tick.decision.reason}",
        f"L={tick.decision.left_duty:+5d}  R={tick.decision.right_duty:+5d}",
        f"State: {tick.decision.state.value}",
        f"Yaw: {np.degrees(tick.pose.yaw_rad):+.1f} deg ({tick.pose.health.value})",
        f"Near-field veto: {'YES' if tick.near_field_veto else 'no'}",
        "",
        f"Stop zone: {'BLOCKED' if tick.stop_zone_blocked else 'clear'} ({tick.stop_zone_count} cells)",
        f"Fwd:  {tick.clearance_forward_m:.2f} m",
        f"Left: {tick.clearance_left_m:.2f} m",
        f"Right:{tick.clearance_right_m:.2f} m",
        "",
        f"Obstacle pts: {tick.n_obstacle_points}",
        f"US: {tick.ultrasonic_m:.2f} m" if tick.ultrasonic_m is not None else "US: n/a",
        "",
        f"Depth: {depth_ms:.1f} ms",
        f"Proj:  {tick.projection_ms:.1f} ms",
        f"Grid:  {tick.grid_ms:.1f} ms",
        f"Total: {tick.total_ms:.1f} ms",
    ]

    y = 16
    for line in lines:
        cv2.putText(img, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        y += 16

    return img


def _compose_panel(
    bgr_frame: np.ndarray,
    depth_color: np.ndarray,
    grid_img: np.ndarray,
    text_img: np.ndarray,
) -> np.ndarray:
    """Arrange the four panels into a 2x2 composite image."""
    # Top row: camera frame | colorized depth.
    top_left = cv2.resize(bgr_frame, (_PANEL_W, _PANEL_H))
    top_right = cv2.resize(depth_color, (_PANEL_W, _PANEL_H))
    top = np.hstack([top_left, top_right])

    # Bottom row: grid (square) | text panel.
    # Pad grid vertically to match _PANEL_H if needed.
    grid_resized = cv2.resize(grid_img, (_PANEL_W, _PANEL_H))
    text_resized = cv2.resize(text_img, (_PANEL_W, _PANEL_H))
    bottom = np.hstack([grid_resized, text_resized])

    return np.vstack([top, bottom])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset through the reactive safety stack.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the recording directory.")
    parser.add_argument("--backend", default=_DEFAULT_DEPTH_BACKEND, help="Depth backend name.")
    parser.add_argument("--extrinsics", type=Path, default=Path(_DEFAULT_EXTRINSICS))
    parser.add_argument("--intrinsics", type=Path, default=Path(_DEFAULT_INTRINSICS))
    parser.add_argument("--save-dir", type=Path, default=None, help="Save annotated frames here.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--headless", action="store_true", help="No imshow window (save-only mode).")
    args = parser.parse_args(argv)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # --- load calibration ---
    intrinsics = load_intrinsics(args.intrinsics)
    extrinsics = load_extrinsics(args.extrinsics)

    # --- load depth backend ---
    print(f"Loading depth backend: {args.backend} ...")
    estimator = build_estimator(args.backend)
    estimator.load()
    print("Depth backend ready.")

    # --- build reactive stack ---
    gyro = GyroIntegrator()
    pose_source = GyroPoseSource(gyro)
    safety = ReactiveSafety(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        pose_source=pose_source,
    )

    # --- iterate events ---
    frame_count = 0
    ultrasonic_m: float | None = None
    print(f"Replaying events from {args.dataset} ...")

    for event in iter_dataset_events(args.dataset):
        payload = event.payload
        if event.channel == "imu" and isinstance(payload, ImuRecord):
            gyro.add_sample(payload.gz, t=payload.t_monotonic)
            continue
        if event.channel == "telemetry" and isinstance(payload, TelemetryRecord):
            raw_cm = payload.payload.get("distance", -1)
            if isinstance(raw_cm, (int, float)) and raw_cm > 0:
                ultrasonic_m = float(raw_cm) / 100.0
            continue
        if event.channel != "frame" or not isinstance(payload, FrameRecord):
            continue

        frame_count += 1
        if args.max_frames is not None and frame_count > args.max_frames:
            break

        bgr = cv2.imread(str(payload.path))
        if bgr is None:
            print(f"  skip {payload.path} (unreadable)")
            continue

        # Depth inference.
        t0 = time.monotonic()
        depth_result = estimator.infer(bgr)
        depth_ms = (time.monotonic() - t0) * 1000.0

        # Reactive tick.
        tick = safety.tick(
            depth_result.depth_m,
            ultrasonic_m=ultrasonic_m,
            t_now=payload.t_monotonic,
        )

        # Visualization.
        depth_color = _colorize_depth(depth_result.depth_m)
        grid_img = _render_grid(safety.grid, tick)
        text_img = _render_text_panel(tick, depth_ms, payload.index)
        panel = _compose_panel(bgr, depth_color, grid_img, text_img)

        if args.save_dir:
            out_path = args.save_dir / f"reactive_{payload.index:06d}.png"
            cv2.imwrite(str(out_path), panel)

        if not args.headless:
            cv2.imshow("Reactive Replay", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                # Pause — wait for another space or q.
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 in (ord(" "), ord("q")):
                        break
                if key2 == ord("q"):
                    break

        if frame_count % 50 == 0:
            print(
                f"  [{frame_count}] depth={depth_ms:.0f}ms "
                f"reactive={tick.total_ms:.1f}ms "
                f"{'BLOCKED' if tick.stop_zone_blocked else 'clear'} "
                f"L={tick.decision.left_duty:+d} R={tick.decision.right_duty:+d}"
            )

    if not args.headless:
        cv2.destroyAllWindows()

    estimator.unload()
    print("Done.")


if __name__ == "__main__":
    main()
