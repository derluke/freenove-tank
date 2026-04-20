"""Rolling-frontier replay regressions against cached observations.

These tests are intentionally loose. They do not try to validate
frontier-planner quality to a specific score; they guard against obvious
collapse modes while the planner is still being tuned:
- straight drives should produce some real frontier-approach commands,
- turn-in-place runs should stay turn-dominant rather than "driving"
  toward phantom targets.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tankbot.desktop.autonomy.dataset import FRAME_LOG_NAME
from tankbot.desktop.autonomy.frontier import PlannerMode, PlannerSnapshot
from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate
from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.replay_frontier import (
    ReplaySummary,
    _discover_datasets,
    _row_for_summary,
    replay,
    sweep_datasets,
)
from tankbot.desktop.autonomy.rolling_frontier import RollingGridConfig
from tankbot.desktop.autonomy.scanmatch_eval import Observations, _cache_path, _load_observations

_BACKEND = "depth_anything_v2"
_SCALE = 1.0


def _cache_for(recording: Path) -> Path:
    return _cache_path(recording, _BACKEND, ProjectionConfig(depth_scale=_SCALE))


def _slice_observations(obs: Observations, n_frames: int) -> Observations:
    n = min(obs.n_frames, n_frames)
    end_offset = int(obs.points_offsets[n])
    return Observations(
        t_monotonic=obs.t_monotonic[:n].copy(),
        frame_idx=obs.frame_idx[:n].copy(),
        gyro_yaw=obs.gyro_yaw[:n].copy(),
        motors_active=obs.motors_active[:n].copy(),
        points_offsets=obs.points_offsets[: n + 1].copy(),
        points_xyz=obs.points_xyz[:end_offset].copy(),
    )


def _summary(
    *,
    final_health: HealthState = HealthState.HEALTHY,
    final_mode: PlannerMode = PlannerMode.APPROACH_FRONTIER,
    final_reason: str = "approach_frontier",
) -> ReplaySummary:
    health_counts = {
        HealthState.HEALTHY: 10,
        HealthState.DEGRADED: 0,
        HealthState.BROKEN: 0,
    }
    if final_health == HealthState.DEGRADED:
        health_counts[HealthState.HEALTHY] = 7
        health_counts[HealthState.DEGRADED] = 3

    return ReplaySummary(
        n_frames=10,
        health_counts=health_counts,
        transitions=[],
        final_pose=PoseEstimate(
            t_monotonic=1.0,
            yaw_rad=0.1,
            yaw_var=0.0,
            xy_m=(1.0, 0.0),
            xy_var=0.0,
            health=final_health,
            source="test",
        ),
        final_command_mode=final_mode,
        final_command_reason=final_reason,
        final_command_target_cell=(60, 70),
        final_selected_frontier_cell=(60, 70),
        final_anchor_xy=(0.0, 0.0),
        final_free_cells=100,
        final_occupied_cells=20,
        final_unknown_cells=50,
        final_frontier_clusters=4,
        total_writes=123,
        recenters=1,
        command_counts={
            PlannerMode.HOLD: 1,
            PlannerMode.TURN: 2,
            PlannerMode.FORWARD: 0,
            PlannerMode.APPROACH_FRONTIER: 7,
            PlannerMode.REVISIT_ANCHOR: 0,
            PlannerMode.RECOVERY_SCAN: 0,
        },
        degraded_continue_ticks=2,
        degraded_hold_ticks=1,
        final_snapshot=PlannerSnapshot(
            planner_mode=final_mode.value,
            frontier_count=4,
            selected_frontier=(1.5, 0.2),
            coverage_ratio=0.4,
            free_cells=100,
            occupied_cells=20,
            unknown_cells=50,
            reason=final_reason,
        ),
    )


def test_discover_datasets_returns_sorted_stream_dirs(tmp_path: Path) -> None:
    root = tmp_path / "phase0c"
    root.mkdir()
    (root / "b_case").mkdir()
    (root / "b_case" / FRAME_LOG_NAME).write_text("", encoding="utf-8")
    (root / "a_case").mkdir()
    (root / "a_case" / FRAME_LOG_NAME).write_text("", encoding="utf-8")
    (root / "straight_2m_raw_concat").mkdir()
    (root / "straight_2m_raw_concat" / FRAME_LOG_NAME).write_text("", encoding="utf-8")
    (root / "ignore_me").mkdir()

    discovered = _discover_datasets(root)

    assert [p.name for p in discovered] == ["a_case", "b_case"]


def test_discover_datasets_can_include_derived_runs(tmp_path: Path) -> None:
    root = tmp_path / "phase0c"
    root.mkdir()
    (root / "a_case").mkdir()
    (root / "a_case" / FRAME_LOG_NAME).write_text("", encoding="utf-8")
    (root / "straight_2m_raw_concat").mkdir()
    (root / "straight_2m_raw_concat" / FRAME_LOG_NAME).write_text("", encoding="utf-8")

    discovered = _discover_datasets(root, include_derived=True)

    assert [p.name for p in discovered] == ["a_case", "straight_2m_raw_concat"]


def test_sweep_datasets_aggregates_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "phase0c"
    root.mkdir()
    datasets = []
    for name in ("straight_case", "round_trip_case"):
        ds = root / name
        ds.mkdir()
        (ds / FRAME_LOG_NAME).write_text("", encoding="utf-8")
        (ds / "ground_truth.json").write_text(
            '{"start_xy_m":[0,0],"end_xy_m":[1,0],"start_yaw_deg":0,"end_yaw_deg":0}',
            encoding="utf-8",
        )
        datasets.append(ds)

    def fake_build_observations(
        dataset: Path,
        *,
        backend: str,
        depth_scale: float,
        intrinsics_path: Path,
        extrinsics_path: Path,
        max_frames: int | None,
        force: bool = False,
        verbose: bool = True,
    ) -> Observations:
        assert backend == _BACKEND
        assert depth_scale == _SCALE
        assert max_frames == 20
        assert not force
        assert verbose
        return Observations(
            t_monotonic=np.array([0.0], dtype=np.float64),
            frame_idx=np.array([0], dtype=np.int32),
            gyro_yaw=np.array([0.0], dtype=np.float64),
            motors_active=np.array([True], dtype=bool),
            points_offsets=np.array([0, 0], dtype=np.int32),
            points_xyz=np.zeros((0, 3), dtype=np.float32),
        )

    def fake_replay(
        obs: Observations,
        *,
        grid_config,
        scan_config=None,
        icp_config=None,
        sm_config=None,
        render_dir=None,
        render_every: int = 10,
        upscale: int = 5,
        progress_every: int = 100,
    ) -> ReplaySummary:
        _ = obs, grid_config, scan_config, icp_config, sm_config, render_dir, render_every, upscale, progress_every
        return _summary()

    monkeypatch.setattr("tankbot.desktop.autonomy.replay_frontier.build_observations", fake_build_observations)
    monkeypatch.setattr("tankbot.desktop.autonomy.replay_frontier.replay", fake_replay)

    rows = sweep_datasets(
        datasets,
        backend=_BACKEND,
        depth_scale=_SCALE,
        intrinsics_path=tmp_path / "intrinsics.yaml",
        extrinsics_path=tmp_path / "extrinsics.yaml",
        max_frames=20,
        rebuild_cache=False,
        grid_config=RollingGridConfig(),
    )

    assert [row.dataset for row in rows] == ["straight_case", "round_trip_case"]
    assert rows[0].final_health == "healthy"
    assert rows[0].final_planner == "approach_frontier"
    assert rows[0].endpoint_error_m == pytest.approx(0.0)
    assert rows[0].heading_error_deg == pytest.approx(math.degrees(0.1))


def test_row_for_summary_omits_ground_truth_metrics_when_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "no_gt"
    dataset.mkdir()

    row = _row_for_summary(dataset, _summary(final_health=HealthState.DEGRADED, final_mode=PlannerMode.TURN, final_reason="pose_degraded"))

    assert row.dataset == "no_gt"
    assert row.endpoint_error_m is None
    assert row.heading_error_deg is None
    assert row.final_health == "degraded"
    assert row.final_planner == "turn"


@pytest.mark.slow
def test_frontier_replay_straight_drive_produces_frontier_commands(phase0c_dir: Path) -> None:
    recording = phase0c_dir / "straight_1m_1"
    cache = _cache_for(recording)
    if not cache.exists():
        pytest.skip(
            f"no observations cache for straight_1m_1 "
            f"(expected {cache.name}). Regenerate with "
            "`task scanmatch:eval DATASET=recordings/phase0c/straight_1m_1`"
        )

    obs = _slice_observations(_load_observations(cache), 180)
    summary = replay(obs, progress_every=0)
    total = max(summary.n_frames, 1)

    assert summary.command_counts[PlannerMode.APPROACH_FRONTIER] > 0, (
        "straight replay never emitted an approach-frontier command — "
        "planner may be collapsing to hold/turn"
    )
    assert summary.command_counts[PlannerMode.HOLD] < 0.2 * total, (
        f"straight replay spent {summary.command_counts[PlannerMode.HOLD] / total:.0%} in HOLD — "
        "planner is overly gated on a healthy straight drive"
    )
    assert summary.final_frontier_clusters > 0, "straight replay ended with no frontier clusters"


@pytest.mark.slow
def test_frontier_replay_turn_in_place_backs_off_frontier_driving(phase0c_dir: Path) -> None:
    recording = phase0c_dir / "turn_in_place_90"
    cache = _cache_for(recording)
    if not cache.exists():
        pytest.skip(
            f"no observations cache for turn_in_place_90 "
            f"(expected {cache.name}). Regenerate with "
            "`task scanmatch:eval DATASET=recordings/phase0c/turn_in_place_90`"
        )

    obs = _slice_observations(_load_observations(cache), 120)
    summary = replay(obs, progress_every=0)
    total = max(summary.n_frames, 1)

    non_frontier = summary.command_counts[PlannerMode.TURN] + summary.command_counts[PlannerMode.HOLD]
    assert summary.health_counts[HealthState.DEGRADED] >= 0.2 * total, (
        "turn-in-place replay barely degraded — pose/health gating may have regressed"
    )
    assert non_frontier >= 0.3 * total, (
        "turn-in-place replay stayed in frontier-driving mode too much — planner "
        "should back off on a rotation-heavy segment"
    )


@pytest.mark.slow
def test_frontier_replay_round_trip_backs_off_when_pose_degrades(phase0c_dir: Path) -> None:
    recording = phase0c_dir / "round_trip_1"
    cache = _cache_for(recording)
    if not cache.exists():
        pytest.skip(
            f"no observations cache for round_trip_1 "
            f"(expected {cache.name}). Regenerate with "
            "`task scanmatch:eval DATASET=recordings/phase0c/round_trip_1`"
        )

    obs = _slice_observations(_load_observations(cache), 240)
    summary = replay(obs, progress_every=0)
    degraded_frames = summary.health_counts.get(HealthState.DEGRADED, 0)

    assert degraded_frames > 0, "round-trip replay never entered DEGRADED; regression no longer exercises the handoff path"
    assert summary.command_counts[PlannerMode.HOLD] >= 0.2 * summary.n_frames, (
        "round-trip replay did not meaningfully back off despite degraded pose"
    )
    assert summary.final_pose.health != HealthState.HEALTHY, (
        "round-trip replay ended fully healthy; expected a hard segment to remain degraded"
    )
