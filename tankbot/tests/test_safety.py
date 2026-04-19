from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tankbot.desktop.autonomy.reactive.grid import ReactiveGrid
from tankbot.desktop.autonomy.reactive.pose import NullPoseSource
from tankbot.desktop.autonomy.reactive.safety import ReactiveSafety


def _synthesize_floor_depth(intrinsics, extrinsics) -> np.ndarray:
    rot = extrinsics.rotation_robot_from_camera()
    trans = extrinsics.translation_robot_from_camera()
    h, w = intrinsics.height, intrinsics.width
    vs, us = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    rays_cam = np.stack(
        [
            (us - intrinsics.cx) / intrinsics.fx,
            (vs - intrinsics.cy) / intrinsics.fy,
            np.ones_like(us),
        ],
        axis=-1,
    )
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_rob = rays_cam @ rot.T
    ts = -trans[2] / rays_rob[..., 2]
    depth = np.full_like(ts, np.nan, dtype=np.float32)
    valid = np.isfinite(ts) & (ts > 0.0) & (ts < 20.0)
    depth[valid] = (ts[valid] * rays_cam[..., 2][valid]).astype(np.float32)
    return depth


def test_near_field_below_floor_veto_does_not_trigger_on_flat_floor(
    calibrated_intrinsics,
    calibrated_extrinsics,
) -> None:
    safety = ReactiveSafety(
        intrinsics=calibrated_intrinsics,
        extrinsics=calibrated_extrinsics,
        pose_source=NullPoseSource(),
        grid=ReactiveGrid(persistence_frames=1),
    )
    depth = _synthesize_floor_depth(calibrated_intrinsics, calibrated_extrinsics)
    assert not safety._near_field_below_floor_veto(depth)


def test_near_field_below_floor_veto_triggers_on_false_far_lower_center_blob(
    calibrated_intrinsics,
    calibrated_extrinsics,
) -> None:
    safety = ReactiveSafety(
        intrinsics=calibrated_intrinsics,
        extrinsics=calibrated_extrinsics,
        pose_source=NullPoseSource(),
        grid=ReactiveGrid(persistence_frames=1),
    )
    h = calibrated_intrinsics.height
    w = calibrated_intrinsics.width
    depth = _synthesize_floor_depth(calibrated_intrinsics, calibrated_extrinsics)
    y0 = int(h * 0.80)
    x0 = int(w * 0.35)
    x1 = int(w * 0.65)
    depth[y0:, x0:x1] = 0.85
    assert safety._near_field_below_floor_veto(depth)


def _build_safety(calibrated_intrinsics, calibrated_extrinsics) -> ReactiveSafety:
    return ReactiveSafety(
        intrinsics=calibrated_intrinsics,
        extrinsics=calibrated_extrinsics,
        pose_source=NullPoseSource(),
        grid=ReactiveGrid(persistence_frames=1),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("image_name", "expected_veto"),
    [
        ("close_fluffy_obstacle.jpg", True),
        ("box_clearance_67cm.jpg", False),
        ("open_floor_clearance_98cm.jpg", False),
        ("open_floor_clearance_98cm_recheck.jpg", False),
    ],
)
def test_live_frame_near_field_veto_regression(
    calibrated_intrinsics,
    calibrated_extrinsics,
    depth_anything_v2_estimator,
    fixtures_dir: Path,
    image_name: str,
    expected_veto: bool,
) -> None:
    image_path = fixtures_dir / "live_reactive" / image_name
    bgr = cv2.imread(str(image_path))
    assert bgr is not None, f"failed to load {image_path}"

    depth = depth_anything_v2_estimator.infer(bgr).depth_m
    safety = _build_safety(calibrated_intrinsics, calibrated_extrinsics)
    assert safety._near_field_below_floor_veto(depth) is expected_veto
