from __future__ import annotations

import math

import numpy as np

from tankbot.desktop.autonomy.reactive.grid import ReactiveGrid


def test_grid_persistence_rotates_old_hits_with_yaw() -> None:
    grid = ReactiveGrid(half_extent_m=1.0, resolution_m=0.1, persistence_frames=2)
    point_ahead = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)

    grid.build_from_points(point_ahead, yaw_rad=0.0)
    grid.build_from_points(np.empty((0, 3), dtype=np.float32), yaw_rad=math.pi / 2)

    # After a +90 deg left turn, a world-fixed obstacle that was ahead
    # should appear on the robot's right in the current robot frame.
    assert grid.clearance_in_direction(-math.pi / 2) <= 0.61
    assert grid.clearance_in_direction(0.0) >= 0.9


def test_grid_persistence_without_rotation_keeps_hits_in_place() -> None:
    grid = ReactiveGrid(half_extent_m=1.0, resolution_m=0.1, persistence_frames=2)
    point_ahead = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)

    grid.build_from_points(point_ahead, yaw_rad=0.0)
    grid.build_from_points(np.empty((0, 3), dtype=np.float32), yaw_rad=0.0)

    assert grid.clearance_in_direction(0.0) <= 0.6
