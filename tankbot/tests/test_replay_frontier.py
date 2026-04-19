"""Rolling-frontier replay regressions against cached observations.

These tests are intentionally loose. They do not try to validate
frontier-planner quality to a specific score; they guard against obvious
collapse modes while the planner is still being tuned:
- straight drives should produce some real frontier-approach commands,
- turn-in-place runs should stay turn-dominant rather than "driving"
  toward phantom targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tankbot.desktop.autonomy.frontier import PlannerMode
from tankbot.desktop.autonomy.reactive.pose import HealthState
from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.replay_frontier import replay
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
def test_frontier_replay_turn_in_place_emits_meaningful_turn_fraction(phase0c_dir: Path) -> None:
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

    assert summary.command_counts[PlannerMode.TURN] >= 0.2 * total, (
        "turn-in-place replay emitted too little TURN behavior — planner may be "
        "over-committing to frontier approach during a rotation-heavy segment"
    )
    assert summary.command_counts[PlannerMode.HOLD] == 0, (
        "turn-in-place replay unexpectedly held position despite healthy pose"
    )


@pytest.mark.slow
def test_frontier_replay_round_trip_uses_bounded_degraded_handoff(phase0c_dir: Path) -> None:
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
    assert summary.degraded_continue_ticks > 0, (
        "degraded replay never continued an in-flight command — "
        "planner regressed to immediate HOLD on every degraded tick"
    )
    assert summary.degraded_continue_ticks < degraded_frames, (
        "degraded replay continued for every degraded tick — "
        "handoff is no longer bounded"
    )
