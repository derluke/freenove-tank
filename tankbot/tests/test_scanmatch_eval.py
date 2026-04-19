from __future__ import annotations

import numpy as np

from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.scanmatch_eval import (
    Observations,
    _cache_path,
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
