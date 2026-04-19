"""Floor-plane projection: depth map → robot-frame 3D points.

Given a metric depth map in meters, camera intrinsics, and the (fixed)
camera extrinsics, produces a point cloud in the **robot frame** suitable
for 2D occupancy gridding.

Conventions:
- Robot frame: +x forward, +y left, +z up. Origin at chassis center at
  floor level.
- Depth values are already metric (the Phase 0b depth backends return
  meters directly).
- A `max_range_m` filter drops far-field points that would fight grid
  resolution and a `min_height_m` filter drops points the projection
  places below the floor (which usually means the depth estimate or the
  pitch calibration is wrong — keep them out of the occupancy grid).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tankbot.desktop.autonomy.reactive.extrinsics import CameraExtrinsics
from tankbot.desktop.autonomy.reactive.intrinsics import CameraIntrinsics


@dataclass(frozen=True)
class ProjectionConfig:
    """Tunables for `project_depth_to_robot_points`."""

    depth_scale: float = 1.0
    """Multiply raw depth by this before anything else.  Monocular
    "metric" models often have a global scale offset.  Calibrate once:
    face a hard flat wall, compare the model's center-strip median with
    the ultrasonic reading, set depth_scale = ultrasonic / median.
    Values < 1 shrink reported distances (model over-estimates)."""

    undistort_pixels: bool = True
    """Convert distorted image pixels to ideal pinhole rays before
    backprojection when calibration includes distortion coefficients."""

    pitch_bias_deg: float = 1.5
    """Small empirical pitch compensation applied at projection time on top
    of the measured camera extrinsics. This keeps physical mount pitch and
    depth/projection-model bias as separate concepts."""

    downsample: int = 4
    """Take every Nth pixel in both axes. 4 -> ~19k points at 640x480."""

    max_range_m: float = 6.0
    """Drop depths beyond this -- they are unreliable for a 640x480 camera
    and they fight the grid resolution."""

    min_range_m: float = 0.15
    """Drop depths below this -- they are usually floor/lens glare."""

    min_height_m: float = -0.05
    """Drop points the projection places below the floor by more than this.
    Negative tolerance accounts for extrinsic-calibration slop."""

    max_height_m: float = 0.45
    """Drop points above this -- ceiling, upper walls, things the tank
    cannot collide with. Keep above the tank's physical height."""

    def cache_tag(self) -> str:
        """Stable filename tag for caches derived from this config."""
        scale = f"s{self.depth_scale:.3f}"
        undist = f"u{1 if self.undistort_pixels else 0}"
        pitch = f"pb{self.pitch_bias_deg:+.2f}".replace("+", "p").replace("-", "m")
        return "_".join([scale, undist, pitch]).replace(".", "p")


def _normalized_camera_xy(
    u: np.ndarray,
    v: np.ndarray,
    intrinsics: CameraIntrinsics,
    cfg: ProjectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal pinhole normalized image coordinates for pixels."""
    if cfg.undistort_pixels and intrinsics.has_distortion:
        import cv2

        points = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float64)
        camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
        undistorted = cv2.undistortPoints(
            points,
            camera_matrix,
            dist_coeffs,
            P=None,
        ).reshape(-1, 2)
        x_norm = undistorted[:, 0].astype(np.float32, copy=False)
        y_norm = undistorted[:, 1].astype(np.float32, copy=False)
        return x_norm, y_norm

    x_norm = ((u - intrinsics.cx) / intrinsics.fx).astype(np.float32, copy=False)
    y_norm = ((v - intrinsics.cy) / intrinsics.fy).astype(np.float32, copy=False)
    return x_norm, y_norm


def _effective_extrinsics(
    extrinsics: CameraExtrinsics,
    cfg: ProjectionConfig,
) -> CameraExtrinsics:
    if cfg.pitch_bias_deg == 0.0:
        return extrinsics
    return replace(extrinsics, pitch_deg=extrinsics.pitch_deg + cfg.pitch_bias_deg)


def project_depth_to_robot_points(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    config: ProjectionConfig | None = None,
) -> np.ndarray:
    """Return an (N, 3) array of robot-frame points (float32, x/y/z in m).

    Invalid pixels — non-finite depth, out of range, below floor, above
    ceiling — are filtered out, so the returned length depends on the
    scene.
    """
    cfg = config or ProjectionConfig()

    if depth_m.ndim != 2:
        msg = f"depth_m must be HxW, got {depth_m.shape}"
        raise ValueError(msg)
    h, w = depth_m.shape

    ds = cfg.downsample if cfg.downsample > 1 else 1
    depth = depth_m[::ds, ::ds]

    # Apply depth scale correction before anything else.
    if cfg.depth_scale != 1.0:
        depth = depth * cfg.depth_scale

    # Pixel coordinates of the downsampled grid, in the ORIGINAL image
    # frame (so intrinsics still apply directly).
    ys, xs = np.meshgrid(
        np.arange(0, h, ds, dtype=np.float32),
        np.arange(0, w, ds, dtype=np.float32),
        indexing="ij",
    )

    finite = np.isfinite(depth)
    in_range = (depth >= cfg.min_range_m) & (depth <= cfg.max_range_m)
    mask = finite & in_range
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)

    z = depth[mask].astype(np.float32)
    u = xs[mask]
    v = ys[mask]

    # Pixel → camera frame. Camera frame is +x right, +y down, +z forward.
    x_norm, y_norm = _normalized_camera_xy(u, v, intrinsics, cfg)
    x_cam = x_norm * z
    y_cam = y_norm * z
    z_cam = z
    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Camera frame → robot frame via the fixed extrinsics.
    effective_extrinsics = _effective_extrinsics(extrinsics, cfg)
    rotation = effective_extrinsics.rotation_robot_from_camera().astype(np.float32)
    translation = effective_extrinsics.translation_robot_from_camera().astype(np.float32)
    p_rob = p_cam @ rotation.T + translation  # (N, 3)

    # Clip on height: drop points below the floor (likely bad depth) and
    # points above the collision-relevant ceiling.
    height_ok = (p_rob[:, 2] >= cfg.min_height_m) & (p_rob[:, 2] <= cfg.max_height_m)
    return p_rob[height_ok].astype(np.float32, copy=False)
