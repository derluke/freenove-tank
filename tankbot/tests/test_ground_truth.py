"""Ground-truth JSON sanity — all phase0c recordings must parse and be within
physically reasonable bounds. Catches things like trailing commas or swapped
start/end fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _ground_truth_dirs(phase0c_dir: Path) -> list[Path]:
    return sorted(
        d for d in phase0c_dir.iterdir() if (d / "ground_truth.json").exists()
    )


def test_at_least_one_ground_truth_recording(phase0c_dir: Path) -> None:
    assert _ground_truth_dirs(phase0c_dir), (
        "no phase0c recording has a ground_truth.json — tests that depend on "
        "ground truth will be silently skipped"
    )


@pytest.mark.parametrize(
    "recording_dir",
    _ground_truth_dirs(Path(__file__).resolve().parent.parent / "recordings" / "phase0c"),
    ids=lambda p: p.name,
)
def test_ground_truth_schema(recording_dir: Path) -> None:
    data = json.loads((recording_dir / "ground_truth.json").read_text(encoding="utf-8"))

    # Required top-level keys.
    for key in ("start_xy_m", "end_xy_m", "tolerance_xy_m", "tolerance_yaw_deg"):
        assert key in data, f"{recording_dir.name}: missing key {key!r}"

    sx, sy = data["start_xy_m"]
    ex, ey = data["end_xy_m"]
    assert all(isinstance(v, (int, float)) for v in (sx, sy, ex, ey))

    tol_xy = float(data["tolerance_xy_m"])
    tol_yaw = float(data["tolerance_yaw_deg"])
    assert 0.01 <= tol_xy <= 1.0, f"{recording_dir.name}: tolerance_xy_m={tol_xy} unrealistic"
    assert 0.5 <= tol_yaw <= 45.0, (
        f"{recording_dir.name}: tolerance_yaw_deg={tol_yaw} unrealistic"
    )

    # Optional start/end yaw default to 0.
    start_yaw = float(data.get("start_yaw_deg", 0.0))
    end_yaw = float(data.get("end_yaw_deg", 0.0))
    assert -360.0 <= start_yaw <= 360.0
    assert -360.0 <= end_yaw <= 360.0
