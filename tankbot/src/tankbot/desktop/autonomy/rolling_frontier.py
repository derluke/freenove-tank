"""Rolling frontier grid — Phase 3 of the autonomy redesign (doc §3.3).

Replacement for the fixed 24x24 m world grid in ``frontier.py``. The
window translates with the robot, cells outside the window are dropped,
and grid writes are gated on pose-health state (doc §3.4).

Scope in this phase:
- Grid maintenance + recentering + pose-health gating only.
- Frontier clustering + target selection + ``PlannerCommand`` output
  reuse the existing contracts from ``frontier.py`` where practical.
- **Offline only.** This module is not wired into live control yet —
  it is exercised from the replay harness. The planner-live switchover
  is Phase 4.

Conventions:
- Robot / world frame: ``+x`` forward, ``+y`` left, ``+z`` up. Matches
  the projection + scan modules.
- Grid indexing: ``cells[row, col]`` where ``col`` is the x axis and
  ``row`` is the y axis. The ``anchor_xy_m`` attribute gives the world
  coordinate of the window **center**. When the robot drifts more than
  ``recenter_margin_m`` from the anchor, the grid recenters — new cells
  that come into the window are unknown, cells that leave are dropped.
- Poses are consumed as ``PoseEstimate`` from ``reactive.pose`` rather
  than raw 4x4 matrices. The grid only trusts ``xy_m != None`` and
  ``health == HEALTHY``; anything else produces no writes, matching
  the pose-health model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from tankbot.desktop.autonomy.frontier import (
    PlannerCommand,
    PlannerMode,
    PlannerSnapshot,
)
from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate

_FREE_HIT_THRESHOLD = 2
_OCC_HIT_THRESHOLD = 2
_MAX_HIT_COUNT = 65535
_MAX_RAY_SAMPLES = 64
_MIN_FRONTIER_CLUSTER = 3
_MIN_OBS_RANGE_M = 0.15
_DEFAULT_MAX_OBS_RANGE_M = 4.0
_DEFAULT_WINDOW_SIZE_M = 12.0
_DEFAULT_RESOLUTION_M = 0.10
_DEFAULT_RECENTER_MARGIN_M = 3.0
# Scan slice thickness (robot-frame z). Matches ScanConfig defaults so
# the grid and the reactive scan pipeline see the same world.
_MIN_HEIGHT_M = 0.02
_MAX_HEIGHT_M = 0.45


@dataclass(frozen=True)
class RollingGridConfig:
    """Tunables for the rolling frontier grid."""

    window_size_m: float = _DEFAULT_WINDOW_SIZE_M
    """Side length of the square window in meters."""

    resolution_m: float = _DEFAULT_RESOLUTION_M
    """Cell edge length in meters. 10 cm = planner-scale detail."""

    recenter_margin_m: float = _DEFAULT_RECENTER_MARGIN_M
    """Recenter the window when the robot is this far from the anchor.
    Recentering is O(grid) so we don't want to do it every tick."""

    max_observation_range_m: float = _DEFAULT_MAX_OBS_RANGE_M
    """Drop depth points farther than this from the robot before
    ray-casting. Monocular depth is unreliable in the far field."""

    min_height_m: float = _MIN_HEIGHT_M
    """Only count points above this height as obstacles. Matches
    ScanConfig so floor returns don't corrupt the map."""

    max_height_m: float = _MAX_HEIGHT_M
    """Only count points below this height as obstacles."""


@dataclass(frozen=True)
class GridSnapshot:
    """What a consumer can see about the current grid state."""

    anchor_xy_m: tuple[float, float]
    window_size_m: float
    resolution_m: float
    free_cells: int
    occupied_cells: int
    unknown_cells: int
    coverage_ratio: float
    last_health: HealthState
    writes_since_reset: int


class RollingFrontierGrid:
    """Window-anchored occupancy grid that follows the robot.

    Responsibilities:
    - maintain ``free_hits`` and ``occ_hits`` arrays in a window
      anchored to ``anchor_xy_m`` (world coords of window center)
    - accept ``update_from_frame(pose, points_robot_xyz)`` and do the
      robot→world transform internally
    - gate writes on ``pose.health == HEALTHY`` and ``pose.xy_m != None``
      per doc §3.4
    - recenter the window when the robot drifts past
      ``recenter_margin_m``; cells that leave the window are dropped,
      cells that enter are initialized to unknown
    """

    def __init__(self, config: RollingGridConfig | None = None) -> None:
        self._cfg = config or RollingGridConfig()
        n = round(self._cfg.window_size_m / self._cfg.resolution_m)
        if n <= 0 or n % 2 != 0:
            msg = f"window_size / resolution must produce an even positive integer, got {n}"
            raise ValueError(msg)
        self._n = n
        self._free_hits = np.zeros((n, n), dtype=np.uint16)
        self._occ_hits = np.zeros((n, n), dtype=np.uint16)
        self._anchor_x: float = 0.0
        self._anchor_y: float = 0.0
        self._last_health: HealthState = HealthState.HEALTHY
        self._writes: int = 0

    # -- public API --------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self._n

    @property
    def anchor_xy_m(self) -> tuple[float, float]:
        return self._anchor_x, self._anchor_y

    @property
    def config(self) -> RollingGridConfig:
        return self._cfg

    def update_from_frame(
        self,
        pose: PoseEstimate,
        points_robot_xyz: np.ndarray,
    ) -> None:
        """Ray-cast ``points_robot_xyz`` from the robot into the grid.

        No-op when the pose is not usable (health != HEALTHY or
        ``xy_m is None``). No-op when the point set is empty.
        """
        self._last_health = pose.health
        if pose.health != HealthState.HEALTHY or pose.xy_m is None:
            return

        # Recenter on every healthy pose — independent of whether this
        # frame contributes point writes. Otherwise a run of empty
        # frames can let the robot drift off the window silently.
        rx, ry = pose.xy_m
        self._maybe_recenter(rx, ry)

        if points_robot_xyz.ndim != 2 or points_robot_xyz.shape[1] != 3:
            return
        if points_robot_xyz.shape[0] == 0:
            return

        # Robot→world transform (2D; ignore z for grid writes).
        cos_y = math.cos(pose.yaw_rad)
        sin_y = math.sin(pose.yaw_rad)
        px = points_robot_xyz[:, 0]
        py = points_robot_xyz[:, 1]
        pz = points_robot_xyz[:, 2]
        wx = rx + cos_y * px - sin_y * py
        wy = ry + sin_y * px + cos_y * py

        height_mask = (pz >= self._cfg.min_height_m) & (pz <= self._cfg.max_height_m)
        dx = wx - rx
        dy = wy - ry
        range2 = dx * dx + dy * dy
        range_mask = (range2 >= _MIN_OBS_RANGE_M * _MIN_OBS_RANGE_M) & (
            range2 <= self._cfg.max_observation_range_m * self._cfg.max_observation_range_m
        )
        valid = height_mask & range_mask & np.isfinite(wx) & np.isfinite(wy)
        wx = wx[valid]
        wy = wy[valid]
        if wx.size == 0:
            return

        start_cell = self._world_to_cell(rx, ry)
        if start_cell is None:
            return

        # Ray-cast point by point — the point count per frame is small
        # (a few thousand at most after projection filtering).
        for i in range(wx.size):
            end_cell = self._world_to_cell(float(wx[i]), float(wy[i]))
            if end_cell is None:
                continue
            ray = self._ray_cells(start_cell, end_cell)
            if len(ray) < 2:
                continue
            for cell in ray[:-1]:
                self._bump(self._free_hits, cell)
            self._bump(self._occ_hits, ray[-1])
            self._writes += 1

    def state_grid(self) -> np.ndarray:
        """0=unknown, 1=free, 2=occupied. Same semantics as frontier.py."""
        state = np.zeros((self._n, self._n), dtype=np.int8)
        free_mask = self._free_hits >= _FREE_HIT_THRESHOLD
        occ_mask = self._occ_hits >= _OCC_HIT_THRESHOLD
        state[free_mask] = 1
        state[occ_mask & (self._occ_hits >= self._free_hits)] = 2
        state[(state == 2) & (self._free_hits > self._occ_hits)] = 1
        return state

    def frontier_clusters(self) -> list[list[tuple[int, int]]]:
        """Find clusters of free cells adjacent to unknown cells."""
        state = self.state_grid()
        free_mask = state == 1
        unknown_mask = state == 0
        frontier_mask = np.zeros_like(free_mask, dtype=bool)
        frontier_mask[1:-1, 1:-1] = free_mask[1:-1, 1:-1] & (
            unknown_mask[:-2, 1:-1]
            | unknown_mask[2:, 1:-1]
            | unknown_mask[1:-1, :-2]
            | unknown_mask[1:-1, 2:]
        )

        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: list[list[tuple[int, int]]] = []
        for idx in np.argwhere(frontier_mask):
            r0, c0 = int(idx[0]), int(idx[1])
            if visited[r0, c0]:
                continue
            cluster: list[tuple[int, int]] = []
            queue = deque([(r0, c0)])
            visited[r0, c0] = True
            while queue:
                cur = queue.popleft()
                cluster.append(cur)
                cr, cc = cur
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (
                        0 <= nr < self._n
                        and 0 <= nc < self._n
                        and frontier_mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if len(cluster) >= _MIN_FRONTIER_CLUSTER:
                clusters.append(cluster)
        return clusters

    def snapshot(self) -> GridSnapshot:
        state = self.state_grid()
        free = int(np.count_nonzero(state == 1))
        occ = int(np.count_nonzero(state == 2))
        unknown = int(np.count_nonzero(state == 0))
        known = free + occ
        total = known + unknown
        return GridSnapshot(
            anchor_xy_m=(self._anchor_x, self._anchor_y),
            window_size_m=self._cfg.window_size_m,
            resolution_m=self._cfg.resolution_m,
            free_cells=free,
            occupied_cells=occ,
            unknown_cells=unknown,
            coverage_ratio=(known / total) if total else 0.0,
            last_health=self._last_health,
            writes_since_reset=self._writes,
        )

    def reset(self) -> None:
        self._free_hits.fill(0)
        self._occ_hits.fill(0)
        self._anchor_x = 0.0
        self._anchor_y = 0.0
        self._writes = 0

    # -- coordinate helpers -----------------------------------------

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Expose for consumers that need to query the grid from world
        coordinates (planner, replay visualizer). Returns ``None``
        when the point is outside the current window."""
        return self._world_to_cell(x, y)

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        res = self._cfg.resolution_m
        half = self._n // 2
        x = self._anchor_x + (col - half) * res
        y = self._anchor_y + (row - half) * res
        return x, y

    # -- internals ---------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        res = self._cfg.resolution_m
        half = self._n // 2
        col = round((x - self._anchor_x) / res) + half
        row = round((y - self._anchor_y) / res) + half
        if 0 <= row < self._n and 0 <= col < self._n:
            return row, col
        return None

    def _maybe_recenter(self, rx: float, ry: float) -> None:
        dx = rx - self._anchor_x
        dy = ry - self._anchor_y
        if math.hypot(dx, dy) <= self._cfg.recenter_margin_m:
            return

        # Snap the anchor to the nearest cell-aligned position so
        # recentering is an integer-cell shift (no resampling).
        res = self._cfg.resolution_m
        shift_col = round(dx / res)
        shift_row = round(dy / res)
        if shift_col == 0 and shift_row == 0:
            return

        self._free_hits = _shift_2d(self._free_hits, shift_row, shift_col)
        self._occ_hits = _shift_2d(self._occ_hits, shift_row, shift_col)
        self._anchor_x += shift_col * res
        self._anchor_y += shift_row * res

    @staticmethod
    def _bump(arr: np.ndarray, cell: tuple[int, int]) -> None:
        if arr[cell] < _MAX_HIT_COUNT:
            arr[cell] = arr[cell] + 1

    @staticmethod
    def _ray_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        sr, sc = start
        er, ec = end
        steps = int(min(_MAX_RAY_SAMPLES, max(abs(er - sr), abs(ec - sc)) + 1))
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for i in range(steps + 1):
            t = i / max(1, steps)
            row = round(sr + (er - sr) * t)
            col = round(sc + (ec - sc) * t)
            cell = (row, col)
            if cell in seen:
                continue
            seen.add(cell)
            cells.append(cell)
        return cells


def _shift_2d(arr: np.ndarray, shift_row: int, shift_col: int) -> np.ndarray:
    """Shift ``arr`` by ``(shift_row, shift_col)`` cells; new cells = 0.

    A positive ``shift_col`` means the window moved +x (east); the data
    shifts left (toward lower cols). Equivalently: the data stays in
    world coordinates, the window origin moves +x, so the grid indexes
    the same world cells at lower (row, col). np.roll + boundary zero
    handles this cleanly.
    """
    out = np.zeros_like(arr)
    n_rows, n_cols = arr.shape

    src_r0 = max(0, shift_row)
    src_r1 = n_rows + min(0, shift_row)
    dst_r0 = max(0, -shift_row)
    dst_r1 = n_rows + min(0, -shift_row)
    src_c0 = max(0, shift_col)
    src_c1 = n_cols + min(0, shift_col)
    dst_c0 = max(0, -shift_col)
    dst_c1 = n_cols + min(0, -shift_col)

    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


__all__ = [
    "GridSnapshot",
    "PlannerCommand",
    "PlannerMode",
    "PlannerSnapshot",
    "RollingFrontierGrid",
    "RollingGridConfig",
]
