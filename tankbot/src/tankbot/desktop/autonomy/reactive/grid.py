"""Robot-centric 2D occupancy grid for the reactive safety layer.

Grid frame = robot frame (+x forward, +y left). The robot sits at
(0, 0). The grid extends `half_extent_m` in every direction from the
origin.

Temporal smoothing: instead of rebuilding from scratch every frame
(which causes visible flicker), the grid keeps a small ring buffer of
recent per-frame hit masks and marks a cell OCCUPIED if it was hit in
*any* of the last `persistence_frames` frames. At 23 Hz with the
default of 3 frames (~130 ms), the robot moves <3 mm between the
oldest and newest frame, so smear from the lack of rotation
compensation is negligible.

Two query methods matter for the reactive layer:
- `stop_zone_occupied(...)` -- is there an obstacle in a forward wedge
  of given distance and half-angle? This is the main safety veto.
- `clearance_in_direction(...)` -- how far can the robot travel along a
  given heading before hitting an obstacle cell? Used by the wander
  policy to pick turn direction when the stop zone is blocked.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Cell state encoding -- small int so the grid stays cheap to copy.
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OCCUPIED = 2


@dataclass(frozen=True)
class StopZoneQuery:
    """Parameters for `ReactiveGrid.stop_zone_occupied`."""

    distance_m: float = 0.35
    """Forward distance the wedge extends. Calibrate against tank
    stopping distance at the cruise speed."""

    half_angle_deg: float = 30.0
    """Half-angle of the forward wedge. 30 deg -> 60 deg total FOV for
    the stop check."""

    min_occupied_cells: int = 2
    """Require at least this many occupied cells before vetoing -- one
    stray cell from depth noise should not halt the robot."""


class ReactiveGrid:
    """Robot-centric occupancy grid with short temporal persistence."""

    def __init__(
        self,
        *,
        half_extent_m: float = 3.0,
        resolution_m: float = 0.05,
        persistence_frames: int = 3,
    ) -> None:
        if resolution_m <= 0:
            msg = f"resolution_m must be positive, got {resolution_m}"
            raise ValueError(msg)
        if half_extent_m <= 0:
            msg = f"half_extent_m must be positive, got {half_extent_m}"
            raise ValueError(msg)
        if persistence_frames < 1:
            msg = f"persistence_frames must be >= 1, got {persistence_frames}"
            raise ValueError(msg)

        self._half_extent_m = float(half_extent_m)
        self._resolution_m = float(resolution_m)
        self._size = round(2 * half_extent_m / resolution_m)
        if self._size <= 0:
            msg = "grid would be empty with given half_extent / resolution"
            raise ValueError(msg)

        self._persistence = persistence_frames

        # Ring buffer of recent per-frame boolean hit masks.
        self._history: deque[np.ndarray] = deque(maxlen=persistence_frames)

        # Cell (row=0, col=0) corresponds to robot-frame (-half, -half).
        # Row index grows with +x (forward). Col index grows with +y (left).
        self._cells = np.zeros((self._size, self._size), dtype=np.uint8)

    @property
    def size(self) -> int:
        return self._size

    @property
    def resolution_m(self) -> float:
        return self._resolution_m

    @property
    def half_extent_m(self) -> float:
        return self._half_extent_m

    @property
    def cells(self) -> np.ndarray:
        return self._cells

    def clear(self) -> None:
        self._cells.fill(CELL_UNKNOWN)
        self._history.clear()

    def build_from_points(self, points_xyz: np.ndarray) -> None:
        """Integrate a new frame of obstacle points into the grid.

        A boolean hit mask for this frame is pushed into the ring buffer.
        The composite grid marks a cell OCCUPIED if it was hit in *any*
        frame still in the buffer, giving short temporal persistence
        without requiring rotation compensation.
        """
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            msg = f"points_xyz must be (N, 3), got {points_xyz.shape}"
            raise ValueError(msg)

        # Build this frame's hit mask.
        hit = np.zeros((self._size, self._size), dtype=np.bool_)
        if points_xyz.shape[0] > 0:
            x = points_xyz[:, 0]
            y = points_xyz[:, 1]
            rows = np.floor((x + self._half_extent_m) / self._resolution_m).astype(np.int32)
            cols = np.floor((y + self._half_extent_m) / self._resolution_m).astype(np.int32)
            inside = (rows >= 0) & (rows < self._size) & (cols >= 0) & (cols < self._size)
            rows = rows[inside]
            cols = cols[inside]
            if rows.size > 0:
                hit[rows, cols] = True

        self._history.append(hit)

        # Rebuild composite: union of all frames in the buffer.
        self._cells.fill(CELL_UNKNOWN)
        for frame_hit in self._history:
            self._cells[frame_hit] = CELL_OCCUPIED

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def stop_zone_occupied(self, query: StopZoneQuery | None = None) -> tuple[bool, int]:
        """Return (occupied?, occupied_cell_count) for the forward wedge."""
        q = query or StopZoneQuery()
        wedge_mask = self._forward_wedge_mask(q.distance_m, q.half_angle_deg)
        occupied = int(((self._cells == CELL_OCCUPIED) & wedge_mask).sum())
        return occupied >= q.min_occupied_cells, occupied

    def clearance_in_direction(self, heading_rad: float, max_distance_m: float | None = None) -> float:
        """Ray-cast from the robot origin along `heading_rad` and return
        the distance to the first CELL_OCCUPIED cell, or the configured
        max (defaults to half_extent_m) if no hit."""
        limit = max_distance_m if max_distance_m is not None else self._half_extent_m
        # Sample the ray at the grid's native resolution.
        steps = math.ceil(limit / self._resolution_m)
        if steps <= 0:
            return limit
        dx = math.cos(heading_rad) * self._resolution_m
        dy = math.sin(heading_rad) * self._resolution_m
        x = 0.0
        y = 0.0
        for i in range(1, steps + 1):
            x += dx
            y += dy
            row = int((x + self._half_extent_m) / self._resolution_m)
            col = int((y + self._half_extent_m) / self._resolution_m)
            if row < 0 or col < 0 or row >= self._size or col >= self._size:
                return i * self._resolution_m
            if self._cells[row, col] == CELL_OCCUPIED:
                return i * self._resolution_m
        return limit

    def _forward_wedge_mask(self, distance_m: float, half_angle_deg: float) -> np.ndarray:
        """Build a boolean mask of cells inside the forward wedge.

        The wedge origin is the robot (+x = forward). Cells are selected
        iff their center's range <= distance_m and |bearing| <= half_angle.
        """
        res = self._resolution_m
        size = self._size
        half = self._half_extent_m
        half_angle = math.radians(half_angle_deg)

        # Cell-center coordinates in robot frame.
        row_idx = np.arange(size, dtype=np.float32)
        col_idx = np.arange(size, dtype=np.float32)
        x_centers = (row_idx + 0.5) * res - half  # shape (size,)
        y_centers = (col_idx + 0.5) * res - half  # shape (size,)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

        # Forward-only: x > 0 (in front of the robot).
        forward = x_grid > 0.0
        range_ok = (x_grid * x_grid + y_grid * y_grid) <= (distance_m * distance_m)
        bearing = np.arctan2(y_grid, x_grid)
        bearing_ok = np.abs(bearing) <= half_angle
        return forward & range_ok & bearing_ok
