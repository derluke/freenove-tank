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

import json
from pathlib import Path

import pytest

from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import ScanMatchConfig
from tankbot.desktop.autonomy.reactive.projection import ProjectionConfig
from tankbot.desktop.autonomy.scanmatch_eval import (
    GroundTruth,
    _cache_path,
    _load_observations,
    evaluate_run,
)

_BACKEND = "depth_anything_v2"
_SCALE = 1.0


def _cache_for(recording: Path) -> Path:
    return _cache_path(recording, _BACKEND, ProjectionConfig(depth_scale=_SCALE))


@pytest.mark.slow
@pytest.mark.parametrize(
    "run_name",
    ["straight_1m_1", "straight_2m", "straight_2m_1"],
)
def test_scanmatch_endpoint_not_catastrophic(phase0c_dir: Path, run_name: str) -> None:
    recording = phase0c_dir / run_name
    cache = _cache_for(recording)
    if not cache.exists():
        pytest.skip(
            f"no observations cache for {run_name} "
            f"(expected {cache.name}). Regenerate with "
            f"`task scanmatch:eval DATASET=recordings/phase0c/{run_name}`"
        )
    gt_path = recording / "ground_truth.json"
    if not gt_path.exists():
        pytest.skip(f"no ground truth for {run_name}")

    obs = _load_observations(cache)
    gt = GroundTruth.from_json(gt_path)
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
def test_round_trip_endpoint_within_tolerance(phase0c_dir: Path) -> None:
    """Round-trip drives return to the start pose (within tolerance).

    Weaker than the straight-drive distance check because the 2x 180 deg
    turns shed pose accuracy, but still a meaningful regression gate.
    """
    recording = phase0c_dir / "round_trip_1"
    cache = _cache_for(recording)
    gt_path = recording / "ground_truth.json"
    if not cache.exists() or not gt_path.exists():
        pytest.skip("missing cache/ground-truth for round_trip_1")

    obs = _load_observations(cache)
    gt = GroundTruth.from_json(gt_path)
    metrics = evaluate_run(
        obs,
        scan_cfg=ScanConfig(),
        icp_cfg=ICPConfig(),
        sm_cfg=ScanMatchConfig(),
        ground_truth=gt,
    )

    gt_tol = json.loads(gt_path.read_text(encoding="utf-8"))
    # Apply 2.5x ground-truth tolerance -- this is a turn-heavy smoke
    # regression, not an accuracy benchmark, and turn drift is not the
    # optimization target for the current projection/motion tuning pass.
    xy_tol = 2.5 * float(gt_tol["tolerance_xy_m"])
    assert metrics.endpoint_error_m < xy_tol, (
        f"round_trip endpoint error {metrics.endpoint_error_m:.3f} m "
        f"exceeds {xy_tol:.3f} m (2.5x tolerance)"
    )
