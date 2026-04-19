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
from typing import Deque

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
_TURN_IN_PLACE_RAD = math.radians(38.0)
_FORWARD_HEADING_RAD = math.radians(55.0)
_FRONTIER_REACHED_M = 0.30
_MIN_TARGET_DISTANCE_M = 0.45
_TARGET_SWITCH_BONUS = 0.75
_REAR_TARGET_PENALTY = 2.4
_DEFER_TARGET_STEPS = 18
_DEGRADED_HANDOFF_MAX_DURATION_S = 0.75
_DEGRADED_HANDOFF_MAX_DISTANCE_M = 0.20
_DEGRADED_FORWARD_GEOMETRY_RAD = math.radians(55.0)
_DEGRADED_TURN_GEOMETRY_RAD = math.radians(100.0)
_TURN_BIAS_WINDOW_S = 0.60
_TURN_BIAS_MIN_RATE_RAD_S = math.radians(8.0)
_ANCHOR_TARGET_SLACK_M = 0.30
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


@dataclass
class _DegradedHandoff:
    target_cell: tuple[int, int]
    mode: PlannerMode
    started_t: float
    origin_xy_m: tuple[float, float]


class RollingFrontierPlanner:
    """Planner-facing wrapper around ``RollingFrontierGrid``.

    This mirrors the coarse frontier selection behavior of the legacy
    fixed-world planner but consumes the rolling window and ``PoseEstimate``
    directly. Write gating stays inside the grid; command gating happens here.

    Conservative behavior:
    - ``HEALTHY`` + ``xy_m`` present: full frontier selection.
    - ``DEGRADED``: may continue the existing in-flight target briefly, but
      never select a new one.
    - ``BROKEN`` or ``xy_m=None``: hold position.
    """

    def __init__(self, config: RollingGridConfig | None = None) -> None:
        self._grid = RollingFrontierGrid(config)
        self._selected_frontier_cell: tuple[int, int] | None = None
        self._selected_frontier_age = 0
        self._deferred_targets: dict[tuple[int, int], int] = {}
        self._tick = 0
        self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="waiting_for_map")
        self._degraded_handoff: _DegradedHandoff | None = None
        self._recent_pose_samples: Deque[tuple[float, float, tuple[float, float]]] = deque()

    @property
    def grid(self) -> RollingFrontierGrid:
        return self._grid

    def defer_target(self, cell: tuple[int, int] | None) -> None:
        if cell is None:
            return
        self._deferred_targets[cell] = self._tick + _DEFER_TARGET_STEPS
        if self._selected_frontier_cell == cell:
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0

    def update_from_frame(self, pose: PoseEstimate, points_robot_xyz: np.ndarray) -> None:
        self._grid.update_from_frame(pose, points_robot_xyz)

    def command_for_pose(self, pose: PoseEstimate) -> PlannerCommand:
        self._tick += 1
        expired = [cell for cell, until in self._deferred_targets.items() if until <= self._tick]
        for cell in expired:
            self._deferred_targets.pop(cell, None)

        if pose.xy_m is None:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="translation_unavailable", confidence=0.0)
            return self._last_command
        if pose.health == HealthState.BROKEN:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_broken", confidence=0.0)
            return self._last_command
        if pose.health == HealthState.DEGRADED:
            return self._command_for_degraded_pose(pose)

        self._degraded_handoff = None
        if pose.health != HealthState.HEALTHY:
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason=f"pose_{pose.health.value}", confidence=0.0)
            return self._last_command
        self._record_pose_sample(pose)
        if self._recent_turn_in_place_motion(pose):
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0
            self._last_command = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=pose.yaw_rad + math.radians(18.0),
                reason="recent_turn_motion",
                confidence=0.2,
            )
            return self._last_command

        frontiers = self._grid.frontier_clusters()
        if not frontiers:
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0
            self._last_command = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=pose.yaw_rad + math.radians(25.0),
                reason="no_frontier_scan",
                confidence=0.2,
            )
            return self._last_command

        selection = self._select_frontier(pose, frontiers)
        if selection is None:
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0
            self._last_command = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=pose.yaw_rad + math.radians(20.0),
                reason="frontier_too_close",
                confidence=0.15,
            )
            return self._last_command

        best_cell, best_score = selection
        if self._selected_frontier_cell == best_cell:
            self._selected_frontier_age += 1
        else:
            self._selected_frontier_cell = best_cell
            self._selected_frontier_age = 1

        target_world = self._grid.cell_to_world(*best_cell)
        dx = target_world[0] - pose.xy_m[0]
        dy = target_world[1] - pose.xy_m[1]
        target_heading = math.atan2(dy, dx)
        heading_error = self._wrap_angle(target_heading - pose.yaw_rad)
        distance = math.hypot(dx, dy)

        if distance <= _FRONTIER_REACHED_M:
            cmd = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=target_heading + math.radians(20.0),
                target_cell=best_cell,
                reason="frontier_reached_reorient",
                confidence=min(1.0, best_score),
            )
        elif abs(heading_error) > _TURN_IN_PLACE_RAD:
            cmd = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=target_heading,
                target_cell=best_cell,
                reason="align_frontier",
                confidence=min(1.0, best_score),
            )
        else:
            cmd = PlannerCommand(
                mode=PlannerMode.APPROACH_FRONTIER if abs(heading_error) < _FORWARD_HEADING_RAD else PlannerMode.FORWARD,
                target_heading=target_heading,
                target_cell=best_cell,
                reason="approach_frontier",
                confidence=min(1.0, best_score),
            )

        self._last_command = cmd
        return cmd

    def _command_for_degraded_pose(self, pose: PoseEstimate) -> PlannerCommand:
        assert pose.xy_m is not None
        if self._degraded_handoff is None:
            self._degraded_handoff = self._begin_degraded_handoff(pose)
        if self._degraded_handoff is None:
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_degraded", confidence=0.0)
            return self._last_command

        handoff = self._degraded_handoff
        if (pose.t_monotonic - handoff.started_t) > _DEGRADED_HANDOFF_MAX_DURATION_S:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_degraded_timeout", confidence=0.0)
            return self._last_command

        if math.hypot(
            pose.xy_m[0] - handoff.origin_xy_m[0],
            pose.xy_m[1] - handoff.origin_xy_m[1],
        ) > _DEGRADED_HANDOFF_MAX_DISTANCE_M:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_degraded_distance_limit", confidence=0.0)
            return self._last_command

        target_world = self._grid.cell_to_world(*handoff.target_cell)
        dx = target_world[0] - pose.xy_m[0]
        dy = target_world[1] - pose.xy_m[1]
        distance = math.hypot(dx, dy)
        if distance <= _FRONTIER_REACHED_M:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_degraded_complete", confidence=0.0)
            return self._last_command

        target_heading = math.atan2(dy, dx)
        heading_error = abs(self._wrap_angle(target_heading - pose.yaw_rad))
        geometry_limit = (
            _DEGRADED_TURN_GEOMETRY_RAD
            if handoff.mode == PlannerMode.TURN
            else _DEGRADED_FORWARD_GEOMETRY_RAD
        )
        if heading_error > geometry_limit:
            self._degraded_handoff = None
            self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="pose_degraded_bad_geometry", confidence=0.0)
            return self._last_command

        self._last_command = PlannerCommand(
            mode=handoff.mode,
            target_heading=target_heading,
            target_cell=handoff.target_cell,
            reason="pose_degraded_continue",
            confidence=min(self._last_command.confidence, 0.35),
        )
        return self._last_command

    def _begin_degraded_handoff(self, pose: PoseEstimate) -> _DegradedHandoff | None:
        assert pose.xy_m is not None
        if self._selected_frontier_cell is None:
            return None
        if self._last_command.mode not in {PlannerMode.TURN, PlannerMode.FORWARD, PlannerMode.APPROACH_FRONTIER}:
            return None
        return _DegradedHandoff(
            target_cell=self._selected_frontier_cell,
            mode=self._last_command.mode,
            started_t=pose.t_monotonic,
            origin_xy_m=pose.xy_m,
        )

    def _record_pose_sample(self, pose: PoseEstimate) -> None:
        assert pose.xy_m is not None
        self._recent_pose_samples.append((pose.t_monotonic, pose.yaw_rad, pose.xy_m))
        cutoff = pose.t_monotonic - _TURN_BIAS_WINDOW_S
        while len(self._recent_pose_samples) >= 2 and self._recent_pose_samples[1][0] < cutoff:
            self._recent_pose_samples.popleft()

    def _recent_turn_in_place_motion(self, pose: PoseEstimate) -> bool:
        if pose.xy_m is None or len(self._recent_pose_samples) < 2:
            return False
        oldest_t, oldest_yaw, _oldest_xy = self._recent_pose_samples[0]
        dt = pose.t_monotonic - oldest_t
        if dt <= 1e-6:
            return False
        yaw_rate = abs(pose.yaw_rad - oldest_yaw) / dt
        return yaw_rate >= _TURN_BIAS_MIN_RATE_RAD_S

    def snapshot(self) -> PlannerSnapshot:
        snap = self._grid.snapshot()
        selected = None
        if self._selected_frontier_cell is not None:
            selected = self._grid.cell_to_world(*self._selected_frontier_cell)
        return PlannerSnapshot(
            planner_mode=self._last_command.mode.value,
            frontier_count=len(self._grid.frontier_clusters()),
            selected_frontier=selected,
            coverage_ratio=snap.coverage_ratio,
            free_cells=snap.free_cells,
            occupied_cells=snap.occupied_cells,
            unknown_cells=snap.unknown_cells,
            reason=self._last_command.reason,
        )

    @property
    def selected_frontier_cell(self) -> tuple[int, int] | None:
        return self._selected_frontier_cell

    def _select_frontier(
        self,
        pose: PoseEstimate,
        clusters: list[list[tuple[int, int]]],
    ) -> tuple[tuple[int, int], float] | None:
        assert pose.xy_m is not None
        cur_x, cur_y = pose.xy_m
        heading = pose.yaw_rad
        anchor_x, anchor_y = self._grid.anchor_xy_m
        pose_anchor_dist = math.hypot(cur_x - anchor_x, cur_y - anchor_y)
        best_cell = clusters[0][0]
        best_score = -1e9
        any_far_target = False
        for cluster in clusters:
            rows = np.array([c[0] for c in cluster], dtype=np.float32)
            cols = np.array([c[1] for c in cluster], dtype=np.float32)
            center_cell = (int(round(rows.mean())), int(round(cols.mean())))
            wx, wy = self._grid.cell_to_world(*center_cell)
            dist = math.hypot(wx - cur_x, wy - cur_y)
            if dist < _MIN_TARGET_DISTANCE_M:
                continue
            target_anchor_dist = math.hypot(wx - anchor_x, wy - anchor_y)
            if target_anchor_dist + _ANCHOR_TARGET_SLACK_M < pose_anchor_dist:
                continue
            any_far_target = True
            target_heading = math.atan2(wy - cur_y, wx - cur_x)
            heading_cost = abs(self._wrap_angle(target_heading - heading))
            info_gain = len(cluster)
            score = (info_gain * 0.35) - (dist * 0.55) - (heading_cost * 0.90)
            if heading_cost < math.radians(35.0):
                score += 0.8
            if heading_cost > math.pi / 2:
                score -= _REAR_TARGET_PENALTY * ((heading_cost - (math.pi / 2)) / (math.pi / 2))
            if center_cell in self._deferred_targets:
                score -= 10.0
            if self._selected_frontier_cell == center_cell:
                score += _TARGET_SWITCH_BONUS
            if score > best_score:
                best_cell = center_cell
                best_score = score
        if not any_far_target:
            return None
        return best_cell, best_score

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle


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
    "RollingFrontierPlanner",
    "RollingGridConfig",
]
