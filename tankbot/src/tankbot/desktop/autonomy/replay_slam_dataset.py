"""Replay a saved image folder through Tankbot's SLAM adapter."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from itertools import chain
from pathlib import Path

import numpy as np

from tankbot.desktop.autonomy.dataset import (
    is_stream_dataset,
    iter_frame_ticks,
    load_manifest,
)
from tankbot.desktop.autonomy.slam import SplatSLAM

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        help="Folder containing either a streaming autonomy dataset or recorded PNG/JPG frames",
    )
    p.add_argument("--save-as", default="robot_replay", help="Basename for logs output")
    p.add_argument("--calib", default="", help="Calibration YAML path")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    frame_ticks = iter_frame_ticks(dataset_dir)
    first_tick = next(frame_ticks, None)
    if first_tick is None:
        raise SystemExit(f"No image frames found in {dataset_dir}")

    manifest = load_manifest(dataset_dir)
    if manifest is not None:
        log.info(
            "Loaded dataset schema v%s from %s",
            manifest.get("schema_version", "unknown"),
            dataset_dir,
        )
    elif not is_stream_dataset(dataset_dir):
        log.info("Loaded legacy image-folder dataset from %s", dataset_dir)

    save_dir = Path("logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{args.save_as}_robot_pipeline.csv"
    ply_path = save_dir / f"{args.save_as}_robot_pipeline.ply"
    pose_path = save_dir / f"{args.save_as}_robot_pipeline.txt"
    upstream_pose_path = save_dir / f"{args.save_as}_robot_pipeline_keyframes.txt"

    slam = SplatSLAM(calibration_path=args.calib)
    slam.load()

    pose_rows: list[tuple[int, np.ndarray]] = []
    base_monotonic = first_tick.frame.t_monotonic
    replay_start = time.monotonic()
    total_frames = int(manifest["counts"]["frames"]) if manifest is not None else None
    replayed_frames = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "frame_t_monotonic",
                "result",
                "tracking_lost",
                "tracking_stable",
                "new_keyframe",
                "num_keyframes",
                "num_points",
                "tracking_quality",
                "pose_x",
                "pose_y",
                "pose_z",
                "telemetry_type",
                "telemetry_distance_cm",
                "telemetry_motor_left",
                "telemetry_motor_right",
            ]
        )

        for tick in chain([first_tick], frame_ticks):
            frame = tick.frame
            frame_bytes = frame.path.read_bytes()
            replay_ts = replay_start + (frame.t_monotonic - base_monotonic)
            result = slam.process_frame(frame_bytes, replay_ts)
            telemetry = tick.telemetry.payload if tick.telemetry is not None else {}
            motor = telemetry.get("motor", {}) if isinstance(telemetry, dict) else {}
            replayed_frames += 1
            if result is None:
                writer.writerow(
                    [
                        frame.index,
                        f"{frame.t_monotonic:.6f}",
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        telemetry.get("type", "") if isinstance(telemetry, dict) else "",
                        telemetry.get("distance", "") if isinstance(telemetry, dict) else "",
                        motor.get("left", "") if isinstance(motor, dict) else "",
                        motor.get("right", "") if isinstance(motor, dict) else "",
                    ]
                )
                continue

            pose = result.camera_pose
            pose_rows.append((frame.index, pose))
            writer.writerow(
                [
                    frame.index,
                    f"{frame.t_monotonic:.6f}",
                    1,
                    int(result.tracking_lost),
                    int(result.tracking_stable),
                    int(result.new_keyframe),
                    result.num_keyframes,
                    result.num_points,
                    f"{result.tracking_quality:.4f}",
                    f"{pose[0, 3]:.6f}",
                    f"{pose[1, 3]:.6f}",
                    f"{pose[2, 3]:.6f}",
                    telemetry.get("type", "") if isinstance(telemetry, dict) else "",
                    telemetry.get("distance", "") if isinstance(telemetry, dict) else "",
                    motor.get("left", "") if isinstance(motor, dict) else "",
                    motor.get("right", "") if isinstance(motor, dict) else "",
                ]
            )

            if frame.index % 50 == 0:
                log.info(
                    "Replay frame %d/%d: lost=%s stable=%s kf=%d points=%d",
                    frame.index,
                    total_frames - 1 if total_frames is not None else frame.index,
                    result.tracking_lost,
                    result.tracking_stable,
                    result.num_keyframes,
                    result.num_points,
                )

    with pose_path.open("w", encoding="utf-8") as f:
        for idx, pose in pose_rows:
            flat = " ".join(f"{x:.9f}" for x in pose[:3, :].reshape(-1))
            f.write(f"{idx} {flat}\n")

    # Save the reconstruction using upstream MASt3R-SLAM evaluation helpers so
    # the output is directly comparable to the vanilla run.
    try:
        from mast3r_slam import evaluate as eval

        timestamps = np.arange(replayed_frames, dtype=np.float32)
        eval.save_traj(
            save_dir,
            upstream_pose_path.name,
            timestamps,
            slam._keyframes,
        )
        eval.save_reconstruction(save_dir, ply_path.name, slam._keyframes, 1.5)
        log.info("Replay PLY saved to %s", ply_path)
        log.info("Replay keyframe trajectory saved to %s", upstream_pose_path)
    except Exception:
        log.exception("Replay completed but upstream-style save failed")

    log.info("Replay trajectory saved to %s", pose_path)
    log.info("Replay CSV saved to %s", csv_path)


if __name__ == "__main__":
    main()
