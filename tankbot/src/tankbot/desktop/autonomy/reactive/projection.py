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

from dataclasses import dataclass

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
    x_cam = (u - intrinsics.cx) * z / intrinsics.fx
    y_cam = (v - intrinsics.cy) * z / intrinsics.fy
    z_cam = z
    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Camera frame → robot frame via the fixed extrinsics.
    rotation = extrinsics.rotation_robot_from_camera().astype(np.float32)
    translation = extrinsics.translation_robot_from_camera().astype(np.float32)
    p_rob = p_cam @ rotation.T + translation  # (N, 3)

    # Clip on height: drop points below the floor (likely bad depth) and
    # points above the collision-relevant ceiling.
    height_ok = (p_rob[:, 2] >= cfg.min_height_m) & (p_rob[:, 2] <= cfg.max_height_m)
    return p_rob[height_ok].astype(np.float32, copy=False)
