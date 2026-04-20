from __future__ import annotations

import json

import numpy as np

from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.dataset import FRAME_LOG_NAME, IMU_LOG_NAME, MANIFEST_NAME
from tankbot.desktop.autonomy.scanmatch_eval import (
    Observations,
    _cache_path,
    _load_observations,
    _save_observations,
    _slice_observations,
    build_observations,
)


def _obs() -> Observations:
    return Observations(
        t_monotonic=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        frame_idx=np.array([0, 1, 2], dtype=np.int32),
        gyro_yaw=np.array([0.0, 0.01, 0.02], dtype=np.float64),
        motors_active=np.array([True, False, True], dtype=bool),
        last_motor_active_t=np.array([0.0, 0.0, 0.2], dtype=np.float64),
        points_offsets=np.array([0, 2, 5, 6], dtype=np.int32),
        points_xyz=np.array(
            [
                [1.0, 0.0, 0.2],
                [1.1, 0.1, 0.2],
                [1.2, 0.0, 0.2],
                [1.3, 0.0, 0.2],
                [1.4, 0.0, 0.2],
                [1.5, 0.0, 0.2],
            ],
            dtype=np.float32,
        ),
    )


def test_slice_observations_truncates_offsets_and_points() -> None:
    sliced = _slice_observations(_obs(), 2)

    assert sliced.n_frames == 2
    assert sliced.frame_idx.tolist() == [0, 1]
    assert sliced.points_offsets.tolist() == [0, 2, 5]
    assert sliced.points_xyz.shape == (5, 3)


def test_build_observations_uses_existing_cache_for_partial_runs(
    monkeypatch,
    tmp_path,
) -> None:
    dataset = tmp_path / "recording"
    dataset.mkdir()
    cache = _cache_path(dataset, "depth_anything_v2", ProjectionConfig(depth_scale=1.0))
    _save_observations(cache, _obs())

    def fail_build_estimator(_backend: str):
        raise AssertionError("build_estimator should not be called when cached observations exist")

    monkeypatch.setattr("tankbot.desktop.autonomy.scanmatch_eval.build_estimator", fail_build_estimator)

    obs = build_observations(
        dataset,
        backend="depth_anything_v2",
        depth_scale=1.0,
        intrinsics_path=tmp_path / "intrinsics.yaml",
        extrinsics_path=tmp_path / "extrinsics.yaml",
        max_frames=2,
        force=False,
        verbose=False,
    )

    assert obs.n_frames == 2
    assert obs.points_offsets.tolist() == [0, 2, 5]


def test_load_observations_hydrates_missing_motor_event_timestamps(tmp_path) -> None:
    dataset = tmp_path / "recording"
    dataset.mkdir()
    (dataset / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (dataset / FRAME_LOG_NAME).write_text(
        json.dumps({"frame_index": 0, "filename": "000000.png", "t_monotonic": 0.10, "t_wall": 0.0})
        + "\n"
        + json.dumps({"frame_index": 1, "filename": "000001.png", "t_monotonic": 0.30, "t_wall": 0.0})
        + "\n",
        encoding="utf-8",
    )
    (dataset / IMU_LOG_NAME).write_text(
        json.dumps({"t_monotonic": 0.05, "t_wall": 0.0, "gz": 0.0, "motor_left": 1200, "motor_right": 1200})
        + "\n"
        + json.dumps({"t_monotonic": 0.08, "t_wall": 0.0, "gz": 0.0, "motor_left": 0, "motor_right": 0})
        + "\n"
        + json.dumps({"t_monotonic": 0.25, "t_wall": 0.0, "gz": 0.0, "motor_left": 1300, "motor_right": 1300})
        + "\n"
        + json.dumps({"t_monotonic": 0.28, "t_wall": 0.0, "gz": 0.0, "motor_left": 0, "motor_right": 0})
        + "\n",
        encoding="utf-8",
    )

    cache = _cache_path(dataset, "depth_anything_v2", ProjectionConfig(depth_scale=1.0))
    np.savez_compressed(
        cache,
        t_monotonic=np.array([0.10, 0.30], dtype=np.float64),
        frame_idx=np.array([0, 1], dtype=np.int32),
        gyro_yaw=np.array([0.0, 0.0], dtype=np.float64),
        motors_active=np.array([False, False], dtype=bool),
        points_offsets=np.array([0, 0, 0], dtype=np.int32),
        points_xyz=np.zeros((0, 3), dtype=np.float32),
    )

    obs = _load_observations(cache)

    assert obs.motors_active.tolist() == [False, False]
    assert obs.last_motor_active_t.tolist() == [0.05, 0.25]
