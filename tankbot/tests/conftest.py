"""Shared fixtures for the tankbot test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankbot.desktop.autonomy.depth import BackendUnavailable, build_estimator
from tankbot.desktop.autonomy.reactive.extrinsics import (
    CameraExtrinsics,
    load_extrinsics,
)
from tankbot.desktop.autonomy.reactive.intrinsics import (
    CameraIntrinsics,
    load_intrinsics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = REPO_ROOT / "calibration"
RECORDINGS_DIR = REPO_ROOT / "recordings" / "phase0c"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def calibrated_intrinsics() -> CameraIntrinsics:
    return load_intrinsics(CALIBRATION_DIR / "camera_intrinsics.yaml")


@pytest.fixture(scope="session")
def calibrated_extrinsics() -> CameraExtrinsics:
    return load_extrinsics(CALIBRATION_DIR / "camera_extrinsics.yaml")


@pytest.fixture(scope="session")
def phase0c_dir() -> Path:
    if not RECORDINGS_DIR.exists():
        pytest.skip(f"Missing recordings dir: {RECORDINGS_DIR}")
    return RECORDINGS_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def depth_anything_v2_estimator():
    try:
        estimator = build_estimator("depth_anything_v2")
        estimator.load()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    yield estimator
    estimator.unload()
