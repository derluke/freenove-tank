"""Geometry correctness for camera extrinsics + depth → robot-frame projection.

These are the regression gates for the class of bug we shipped through
Phase 0c: wrong camera height / pitch / depth_scale silently turning
every floor pixel into a phantom obstacle. The tests are cheap (pure
numpy) and use the real calibration files.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from tankbot.desktop.autonomy.projection_inspect import project_per_pixel
from tankbot.desktop.autonomy.reactive.extrinsics import CameraExtrinsics
from tankbot.desktop.autonomy.reactive.intrinsics import CameraIntrinsics
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
    project_depth_to_robot_points,
)
from tankbot.desktop.autonomy.reactive.scan import ScanConfig


def _ray_angle_below_horizontal_deg(
    intr: CameraIntrinsics, extr: CameraExtrinsics, v_pixel: float
) -> float:
    """Physical downward angle of the ray through pixel row `v_pixel`."""
    rot = extr.rotation_robot_from_camera()
    pixel_ray_cam = np.array(
        [0.0, (v_pixel - intr.cy) / intr.fy, 1.0], dtype=np.float64
    )
    pixel_ray_cam /= np.linalg.norm(pixel_ray_cam)
    pixel_ray_rob = rot @ pixel_ray_cam
    # x_rob is forward, z_rob is up. Downward angle = -asin(z_rob).
    return math.degrees(-math.asin(pixel_ray_rob[2]))


def _floor_ray_depth(
    intr: CameraIntrinsics, extr: CameraExtrinsics, v_pixel: float
) -> float:
    """Depth along ray through pixel (cx, v_pixel) that hits the floor."""
    rot = extr.rotation_robot_from_camera()
    trans = extr.translation_robot_from_camera()
    ray_cam = np.array([0.0, (v_pixel - intr.cy) / intr.fy, 1.0], dtype=np.float64)
    ray_rob = rot @ ray_cam
    # trans.z + t * ray_rob.z = 0  →  t = -trans.z / ray_rob.z
    return float(-trans[2] / ray_rob[2])


class TestExtrinsicsAxisBasics:
    """Sanity checks on the rotation matrix without depending on pitch."""

    def test_forward_optical_axis_maps_mostly_forward(
        self, calibrated_extrinsics: CameraExtrinsics
    ) -> None:
        rot = calibrated_extrinsics.rotation_robot_from_camera()
        forward_cam = np.array([0.0, 0.0, 1.0])
        forward_rob = rot @ forward_cam
        assert forward_rob[0] > 0.9, "optical axis should map close to +x_rob"
        assert abs(forward_rob[1]) < 0.01, "no yaw → no lateral component"

    def test_rotation_is_proper_orthonormal(
        self, calibrated_extrinsics: CameraExtrinsics
    ) -> None:
        rot = calibrated_extrinsics.rotation_robot_from_camera()
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)


class TestFloorProjection:
    """A true floor point, viewed from the calibrated camera, must land at z_rob≈0."""

    @pytest.mark.parametrize("v_offset_from_cy", [60, 90, 120, 180])
    def test_true_floor_projects_to_floor(
        self,
        calibrated_intrinsics: CameraIntrinsics,
        calibrated_extrinsics: CameraExtrinsics,
        v_offset_from_cy: int,
    ) -> None:
        intr = calibrated_intrinsics
        extr = calibrated_extrinsics
        v = intr.cy + v_offset_from_cy
        # Compute depth along the center-column ray that exactly hits the floor,
        # then synthesize a single-pixel HxW depth map and project it.
        d = _floor_ray_depth(intr, extr, v)
        if d <= 0 or not math.isfinite(d):
            pytest.skip(f"ray through v={v} does not hit floor")
        depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
        u = round(intr.cx)
        depth[round(v), u] = d

        cfg = ProjectionConfig(
            depth_scale=1.0,
            downsample=1,
            min_height_m=-0.5,
            max_height_m=1.0,
            undistort_pixels=False,
            pitch_bias_deg=0.0,
        )
        pts = project_depth_to_robot_points(depth, intr, extr, cfg)
        assert pts.shape == (1, 3)
        # z_rob within a millimeter of floor — this is the algebraic identity.
        assert abs(pts[0, 2]) < 1e-3, f"expected floor, got z_rob={pts[0, 2]:.4f} m"

    def test_depth_scale_lt_1_pushes_floor_into_scan_band(
        self,
        calibrated_intrinsics: CameraIntrinsics,
        calibrated_extrinsics: CameraExtrinsics,
    ) -> None:
        """Regression guard for the 0.56-scale bug.

        With the correct scale (1.0) the floor projects to z_rob≈0. With
        a bogus 0.56 scale applied on top of already-metric depth, the
        same floor point pops up into the [0.02, 0.35] scan band. The
        assertion here is: don't regress the default back to 0.56.
        """
        intr = calibrated_intrinsics
        extr = calibrated_extrinsics
        v = intr.cy + 90  # well below principal point
        d_true = _floor_ray_depth(intr, extr, v)
        if d_true <= 0:
            pytest.skip("ray does not hit floor")

        depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
        depth[round(v), round(intr.cx)] = d_true

        cfg_ok = ProjectionConfig(
            depth_scale=1.0,
            downsample=1,
            min_height_m=-0.5,
            undistort_pixels=False,
            pitch_bias_deg=0.0,
        )
        cfg_bad = ProjectionConfig(
            depth_scale=0.56,
            downsample=1,
            min_height_m=-0.5,
            undistort_pixels=False,
            pitch_bias_deg=0.0,
        )

        pts_ok = project_depth_to_robot_points(depth, intr, extr, cfg_ok)
        pts_bad = project_depth_to_robot_points(depth, intr, extr, cfg_bad)
        assert pts_ok.shape == (1, 3)
        assert pts_bad.shape == (1, 3)

        scan = ScanConfig()
        in_band_ok = scan.min_height_m <= pts_ok[0, 2] <= scan.max_height_m
        in_band_bad = scan.min_height_m <= pts_bad[0, 2] <= scan.max_height_m
        assert not in_band_ok, "scale=1.0 should leave floor below scan band"
        assert in_band_bad, (
            "scale=0.56 is expected to wrongly push floor into scan band — "
            "if this suddenly passes, depth_scale defaults or projection "
            "semantics may have drifted."
        )


class TestSyntheticFloorImage:
    """End-to-end per-pixel projection on a synthetic full-floor depth map.

    Rasterises a perfectly flat floor under the calibrated camera and
    asserts that the per-pixel classifier puts ~0 pixels in the scan band
    for the lower half of the image. This is the bug signature we just
    debugged with the `projection:inspect` tool and matches its console
    output semantics.
    """

    @staticmethod
    def _synthesize_floor_depth(
        intr: CameraIntrinsics, extr: CameraExtrinsics
    ) -> np.ndarray:
        rot = extr.rotation_robot_from_camera()
        trans = extr.translation_robot_from_camera()
        h, w = intr.height, intr.width
        vs, us = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64),
            indexing="ij",
        )
        rays_cam = np.stack(
            [
                (us - intr.cx) / intr.fx,
                (vs - intr.cy) / intr.fy,
                np.ones_like(us),
            ],
            axis=-1,
        )
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
        rays_rob = rays_cam @ rot.T
        # Parametric line: p(t) = trans + t * rays_rob. Hits floor when z=0.
        ts = -trans[2] / rays_rob[..., 2]
        depth = np.full_like(ts, np.nan)
        valid = np.isfinite(ts) & (ts > 0.0) & (ts < 20.0)
        # The depth (z_cam) is t * (rays_cam . z_cam) = t * rays_cam[z]; since
        # rays_cam is unit and z_cam = 1 before normalization, use the
        # unnormalized-z-axis projection.
        unit_ray_z = rays_cam[..., 2]
        depth[valid] = (ts[valid] * unit_ray_z[valid]).astype(np.float32)
        return depth.astype(np.float32)

    def test_synthetic_flat_floor_puts_nothing_in_scan_band(
        self,
        calibrated_intrinsics: CameraIntrinsics,
        calibrated_extrinsics: CameraExtrinsics,
    ) -> None:
        depth = self._synthesize_floor_depth(
            calibrated_intrinsics, calibrated_extrinsics
        )
        proj = project_per_pixel(
            depth,
            calibrated_intrinsics,
            calibrated_extrinsics,
            ProjectionConfig(
                depth_scale=1.0,
                undistort_pixels=False,
                pitch_bias_deg=0.0,
            ),
            ScanConfig(),
        )
        total_valid = int((proj.classification != 0).sum())
        band = int((proj.classification == 2).sum())
        band_frac = band / max(total_valid, 1)
        # A perfectly flat synthetic floor must have near-zero scan-band pixels.
        # Allow a small epsilon for pixels right at the horizon row where the
        # ray barely grazes the floor.
        assert band_frac < 0.01, (
            f"synthetic flat floor produced {band_frac:.1%} scan-band pixels — "
            "either extrinsics, depth_scale, or projection math has drifted"
        )


class TestHorizonGeometry:
    """The calibrated horizon-row geometry should match the documented setup."""

    def test_horizon_row_matches_pitch(
        self,
        calibrated_intrinsics: CameraIntrinsics,
        calibrated_extrinsics: CameraExtrinsics,
    ) -> None:
        """A pixel at the principal point points along the optical axis.

        Its downward angle below horizontal should equal the extrinsic
        pitch. This locks the sign convention: pitch_deg>0 = tilted DOWN.
        """
        angle = _ray_angle_below_horizontal_deg(
            calibrated_intrinsics, calibrated_extrinsics, calibrated_intrinsics.cy
        )
        assert abs(angle - calibrated_extrinsics.pitch_deg) < 0.1


class TestDistortionAwareProjection:
    """Real calibration includes lens distortion, so projection must undo it."""

    def test_undistort_pixels_recovers_off_axis_floor_point(
        self,
        calibrated_intrinsics: CameraIntrinsics,
        calibrated_extrinsics: CameraExtrinsics,
    ) -> None:
        intr = calibrated_intrinsics
        extr = calibrated_extrinsics
        camera_matrix = np.array(
            [
                [intr.fx, 0.0, intr.cx],
                [0.0, intr.fy, intr.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.asarray(intr.dist_coeffs, dtype=np.float64)

        # Pick an off-axis floor point so lens distortion matters.
        target_rob = np.array([1.4, 0.6, 0.0], dtype=np.float64)
        rot = extr.rotation_robot_from_camera()
        trans = extr.translation_robot_from_camera()
        target_cam = rot.T @ (target_rob - trans)
        assert target_cam[2] > 0.0

        px, _ = cv2.projectPoints(
            target_cam.reshape(1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        u, v = px.reshape(2)
        ui = int(round(u))
        vi = int(round(v))
        if not (0 <= ui < intr.width and 0 <= vi < intr.height):
            pytest.skip("projected test point fell outside the image")

        depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
        depth[vi, ui] = np.float32(target_cam[2])

        base_cfg = ProjectionConfig(
            depth_scale=1.0,
            downsample=1,
            min_height_m=-1.0,
            max_height_m=1.0,
            undistort_pixels=False,
            pitch_bias_deg=0.0,
        )
        undist_cfg = ProjectionConfig(
            depth_scale=1.0,
            downsample=1,
            min_height_m=-1.0,
            max_height_m=1.0,
            undistort_pixels=True,
            pitch_bias_deg=0.0,
        )

        pts_base = project_depth_to_robot_points(depth, intr, extr, base_cfg)
        pts_undist = project_depth_to_robot_points(depth, intr, extr, undist_cfg)
        assert pts_base.shape == (1, 3)
        assert pts_undist.shape == (1, 3)

        err_base = float(np.linalg.norm(pts_base[0] - target_rob))
        err_undist = float(np.linalg.norm(pts_undist[0] - target_rob))
        assert err_undist < 0.02
        assert err_base > err_undist + 0.01
