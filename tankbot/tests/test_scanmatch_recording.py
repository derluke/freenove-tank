"""End-to-end scan-match regression against cached recordings.

These tests replay a recording through the scan-match pipeline and
check the endpoint against the tape-measured ground truth. They depend
on a pre-built observations cache (produced by `task scanmatch:eval`)
so they don't run the GPU depth model — if the cache is missing they
skip with a clear message.

Scope: this is a *regression* guard, not a tuning target. The bar is
deliberately loose (no catastrophic breakage) so we notice when an
unrelated change knocks out tracking, without making the test brittle
to routine config tweaks.
"""

from __future__ import annotations
from pathlib import Path

import pytest

from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import ScanMatchConfig
from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.scanmatch_eval import (
    GroundTruth,
    Observations,
    _cache_path,
    _load_observations,
    _slice_observations,
    evaluate_run,
)

_BACKEND = "depth_anything_v2"
_SCALE = 1.0
_FRAME_LIMITS = {
    # After ~400 frames this recording is just parked in front of a wall.
    # Keep the smoke test focused on the informative motion segment.
    "straight_2m": 400,
}


def _cache_for(recording: Path) -> Path:
    return _cache_path(recording, _BACKEND, ProjectionConfig(depth_scale=_SCALE))


def _observations_for_run(recording: Path) -> tuple[GroundTruth, Observations]:
    cache = _cache_for(recording)
    gt_path = recording / "ground_truth.json"
    if not cache.exists():
        pytest.skip(
            f"no observations cache for {recording.name} "
            f"(expected {cache.name}). Regenerate with "
            f"`task scanmatch:eval DATASET=recordings/phase0c/{recording.name}`"
        )
    if not gt_path.exists():
        pytest.skip(f"no ground truth for {recording.name}")

    obs = _load_observations(cache)
    obs = _slice_observations(obs, _FRAME_LIMITS.get(recording.name))
    gt = GroundTruth.from_json(gt_path)
    return gt, obs


@pytest.mark.slow
@pytest.mark.parametrize(
    "run_name",
    ["straight_1m_1", "straight_2m", "straight_2m_1"],
)
def test_scanmatch_endpoint_not_catastrophic(phase0c_dir: Path, run_name: str) -> None:
    recording = phase0c_dir / run_name
    gt, obs = _observations_for_run(recording)
    metrics = evaluate_run(
        obs,
        scan_cfg=ScanConfig(),
        icp_cfg=ICPConfig(),
        sm_cfg=ScanMatchConfig(),
        ground_truth=gt,
    )

    true_distance = abs(gt.end_xy_m[0] - gt.start_xy_m[0])

    # Regression guards (loose on purpose — we want "something broke" signal,
    # not an exact-numbers test):
    # 1) The pipeline must reach at least 25% of the true distance. The
    #    pre-fix failure mode was max_x_reached ~ 2 mm on a 1-2 m drive.
    assert metrics.max_x_reached > 0.25 * true_distance, (
        f"{run_name}: max_x_reached={metrics.max_x_reached:.3f} m for a "
        f"{true_distance:.2f} m drive — pipeline appears collapsed"
    )

    # 2) Less than half the run should be BROKEN. The pre-fix failure mode
    #    hit 90% broken on straight drives.
    assert metrics.frac_broken < 0.5, (
        f"{run_name}: frac_broken={metrics.frac_broken:.0%} — health "
        "gating is falsely demoting the pose source"
    )


@pytest.mark.slow
def test_round_trip_does_not_spend_majority_broken(phase0c_dir: Path) -> None:
    """Turn-heavy round trips are a smoke test, not an accuracy target.

    Long-horizon return-to-origin behavior still depends on capabilities
    we have not implemented yet (loop closure / frontier extension /
    stronger translation). The useful regression signal here is simply
    that the pose source does not collapse into BROKEN for most of the run.
    """
    recording = phase0c_dir / "round_trip_1"
    gt, obs = _observations_for_run(recording)
    metrics = evaluate_run(
        obs,
        scan_cfg=ScanConfig(),
        icp_cfg=ICPConfig(),
        sm_cfg=ScanMatchConfig(),
        ground_truth=gt,
    )

    assert metrics.frac_broken < 0.5, (
        f"round_trip spent {metrics.frac_broken:.0%} broken — pose source appears collapsed"
    )
    assert metrics.frac_xy_valid > 0.5, (
        f"round_trip kept xy valid for only {metrics.frac_xy_valid:.0%} of frames"
    )


@pytest.mark.slow
def test_turn_in_place_stays_near_origin(phase0c_dir: Path) -> None:
    """Pure rotation should not fabricate large translation."""
    recording = phase0c_dir / "turn_in_place_90"
    gt, obs = _observations_for_run(recording)
    metrics = evaluate_run(
        obs,
        scan_cfg=ScanConfig(),
        icp_cfg=ICPConfig(),
        sm_cfg=ScanMatchConfig(),
        ground_truth=gt,
    )

    assert metrics.endpoint_error_m < 0.50, (
        f"turn_in_place fabricated {metrics.endpoint_error_m:.3f} m of translation"
    )
