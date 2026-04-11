"""Replay a saved image folder through Tankbot's SLAM adapter."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

from tankbot.desktop.autonomy.slam import SplatSLAM


log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Folder containing recorded PNG/JPG frames")
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

    frame_paths = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frame_paths:
        raise SystemExit(f"No image frames found in {dataset_dir}")

    save_dir = Path("logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{args.save_as}_robot_pipeline.csv"
    ply_path = save_dir / f"{args.save_as}_robot_pipeline.ply"
    pose_path = save_dir / f"{args.save_as}_robot_pipeline.txt"
    upstream_pose_path = save_dir / f"{args.save_as}_robot_pipeline_keyframes.txt"

    slam = SplatSLAM(calibration_path=args.calib)
    slam.load()

    pose_rows: list[tuple[int, np.ndarray]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
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
            ]
        )

        for idx, frame_path in enumerate(frame_paths):
            frame_bytes = frame_path.read_bytes()
            result = slam.process_frame(frame_bytes, time.monotonic())
            if result is None:
                writer.writerow([idx, 0, "", "", "", "", "", "", "", "", ""])
                continue

            pose = result.camera_pose
            pose_rows.append((idx, pose))
            writer.writerow(
                [
                    idx,
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
                ]
            )

            if idx % 50 == 0:
                log.info(
                    "Replay frame %d/%d: lost=%s stable=%s kf=%d points=%d",
                    idx,
                    len(frame_paths) - 1,
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

        timestamps = np.arange(len(frame_paths), dtype=np.float32)
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
