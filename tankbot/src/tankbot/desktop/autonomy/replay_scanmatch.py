"""Replay a recorded dataset through the Phase 2 scan-match pipeline.

Iterates `frame`, `telemetry`, and `imu` events in recorded monotonic
time order so `ScanMatchPoseSource` sees exactly the inputs the live
engine saw:

- **IMU gz** feeds `GyroIntegrator`; the integrated yaw is passed to
  `update(gyro_yaw=...)` as the authoritative rotation prior.
- **Motor state** (from the IMU piggyback when present, falling back
  to the 1 Hz telemetry `motor` field) gates `motors_active` — the
  pose is frozen while the robot is stationary so scan-match noise
  doesn't drift the map.

Datasets recorded before the IMU/motor channels landed still replay —
they fall back to `gyro_yaw=None` and `motors_active=True`, matching
the pre-Phase-2 behavior.

Usage:
    tankbot-replay-scanmatch --dataset recordings/mast3r_raw/vanilla1
    tankbot-replay-scanmatch --dataset <path> --save-dir logs/scanmatch_replay
"""

from __future__ import annotations

import argparse
import math
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
from tankbot.desktop.autonomy.reactive import load_extrinsics, load_intrinsics
from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator
from tankbot.desktop.autonomy.reactive.pose import HealthState
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
    gyro_yaw_deg: float | None,
    motors_active: bool,
) -> np.ndarray:
    h, w = _PANEL_H, _PANEL_W
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
        f"Gyro: {gyro_yaw_deg:.1f} deg" if gyro_yaw_deg is not None else "Gyro: n/a",
        f"Motors: {'ACTIVE' if motors_active else 'idle'}",
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
        help="Depth scale correction (measured_m / depth_model_median).",
    )
    parser.add_argument(
        "--ignore-imu",
        action="store_true",
        help="Ignore imu.jsonl and let scan-match solve rotation itself (pre-Phase-2 behavior).",
    )
    parser.add_argument(
        "--assume-motors-active",
        action="store_true",
        help="Treat motors as always active. Useful on datasets without a motor channel.",
    )
    args = parser.parse_args(argv)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics(args.intrinsics)
    extrinsics = load_extrinsics(args.extrinsics)
    proj_cfg = ProjectionConfig(depth_scale=args.depth_scale)

    print(f"Loading depth backend: {args.backend} ...")
    estimator = build_estimator(args.backend)
    estimator.load()
    print("Depth backend ready.")

    pose_source = ScanMatchPoseSource(
        scan_config=ScanConfig(),
        icp_config=ICPConfig(),
        config=ScanMatchConfig(),
    )
    world_map = WorldMap()
    gyro = GyroIntegrator()

    # Runtime state that updates as non-frame events arrive.
    motors_active = bool(args.assume_motors_active)
    imu_seen = False
    motor_seen = args.assume_motors_active
    frame_count = 0
    n_events = 0

    def _on_telemetry(rec: TelemetryRecord) -> None:
        nonlocal motors_active, motor_seen
        motor = rec.payload.get("motor")
        if isinstance(motor, dict):
            left = int(motor.get("left", 0))
            right = int(motor.get("right", 0))
            motors_active = left != 0 or right != 0
            motor_seen = True

    def _on_imu(rec: ImuRecord) -> None:
        nonlocal motors_active, imu_seen, motor_seen
        gyro.add_sample(rec.gz, t=rec.t_monotonic)
        imu_seen = True
        if rec.motor_left is not None and rec.motor_right is not None:
            motors_active = rec.motor_left != 0 or rec.motor_right != 0
            motor_seen = True

    def _on_frame(rec: FrameRecord) -> bool:
        """Run one pipeline step for this frame. Returns False to abort."""
        nonlocal frame_count
        bgr = cv2.imread(str(rec.path))
        if bgr is None:
            print(f"  skip {rec.path} (unreadable)")
            return True

        t0 = time.monotonic()
        depth_result = estimator.infer(bgr)
        depth_ms = (time.monotonic() - t0) * 1000.0

        points = project_depth_to_robot_points(
            depth_result.depth_m,
            intrinsics,
            extrinsics,
            proj_cfg,
        )

        gyro_yaw = None if args.ignore_imu or not imu_seen else gyro.yaw_rad
        effective_motors = motors_active or not motor_seen

        t_match = time.monotonic()
        pose_source.update(
            points,
            gyro_yaw=gyro_yaw,
            motors_active=effective_motors,
            t_monotonic=rec.t_monotonic,
        )
        match_ms = (time.monotonic() - t_match) * 1000.0

        pose = pose_source.latest()
        rx, ry = pose.xy_m if pose.xy_m is not None else (0.0, 0.0)
        ryaw = pose.yaw_rad

        good = pose.health == HealthState.HEALTHY and pose.xy_m is not None
        world_map.add_frame(rx, ry, ryaw, points, good_pose=good)

        depth_color = _colorize_depth(depth_result.depth_m)
        map_img = world_map.render()
        stats_img = _render_stats(
            pose_source,
            depth_ms,
            match_ms,
            rec.index,
            points.shape[0],
            math.degrees(gyro.yaw_rad) if imu_seen else None,
            effective_motors,
        )

        top_left = cv2.resize(bgr, (_PANEL_W, _PANEL_H))
        top_right = cv2.resize(depth_color, (_PANEL_W, _PANEL_H))
        top = np.hstack([top_left, top_right])
        map_resized = cv2.resize(map_img, (_PANEL_W, _PANEL_H))
        bottom = np.hstack([map_resized, stats_img])
        panel = np.vstack([top, bottom])

        if args.save_dir:
            out_path = args.save_dir / f"scanmatch_{rec.index:06d}.png"
            cv2.imwrite(str(out_path), panel)

        if not args.headless:
            cv2.imshow("Scan Match Replay", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 in (ord(" "), ord("q")):
                        break
                if key2 == ord("q"):
                    return False

        frame_count += 1
        if frame_count % 50 == 0:
            match = pose_source.last_match
            res_str = f"res={match.mean_residual:.4f}" if match else "first"
            gy = f"{math.degrees(gyro.yaw_rad):+.1f}°" if imu_seen else "n/a"
            mt = "on" if effective_motors else "off"
            print(
                f"  [{frame_count}] depth={depth_ms:.0f}ms match={match_ms:.1f}ms "
                f"pose=({rx:+.3f},{ry:+.3f}) yaw={math.degrees(ryaw):+.1f}° "
                f"gyro={gy} mot={mt} {pose.health.value} {res_str}"
            )
        return True

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Replaying events from {args.dataset} ...")
    try:
        for event in iter_dataset_events(args.dataset):
            n_events += 1
            if event.channel == "telemetry" and isinstance(event.payload, TelemetryRecord):
                _on_telemetry(event.payload)
            elif event.channel == "imu" and isinstance(event.payload, ImuRecord):
                _on_imu(event.payload)
            elif event.channel == "frame" and isinstance(event.payload, FrameRecord):
                if not _on_frame(event.payload):
                    break
                if args.max_frames and frame_count >= args.max_frames:
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        estimator.unload()

    if frame_count == 0:
        print(f"No frames found in {args.dataset}", file=sys.stderr)
        sys.exit(1)

    pose = pose_source.latest()
    final_xy = pose.xy_m if pose.xy_m is not None else (0.0, 0.0)
    print()
    print(f"Events:         {n_events}")
    print(f"Frames:         {frame_count}")
    print(f"IMU seen:       {imu_seen}")
    print(f"Motor seen:     {motor_seen}")
    print(f"Final pose:     ({final_xy[0]:+.3f}, {final_xy[1]:+.3f}) m")
    print(f"Final yaw:      {math.degrees(pose.yaw_rad):+.2f} deg")
    if imu_seen:
        print(f"Gyro yaw:       {math.degrees(gyro.yaw_rad):+.2f} deg")
    print(f"Final health:   {pose.health.value}")


if __name__ == "__main__":
    main()
