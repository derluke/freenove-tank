"""Coarse occupancy/frontier planning for room exploration.

This module sits between dense SLAM and low-level motor control.
It maintains a conservative 2D planning grid on the room floor plane
and emits short-horizon commands toward frontiers or anchor views.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

GRID_RESOLUTION_M = 0.10
GRID_SIZE = 240
GRID_CENTER = GRID_SIZE // 2
MAX_RAY_SAMPLES = 48
MAX_OBSERVATION_RANGE_M = 4.5
MIN_FRONTIER_CLUSTER = 3
TURN_IN_PLACE_RAD = math.radians(38.0)
FORWARD_HEADING_RAD = math.radians(55.0)
FRONTIER_REACHED_M = 0.30
MIN_TARGET_DISTANCE_M = 0.45
TRACKING_UNSAFE_SCORE = 0.35
TARGET_SWITCH_BONUS = 0.75
REAR_TARGET_PENALTY = 2.4
DEFER_TARGET_STEPS = 18


class PlannerMode(str, Enum):
    HOLD = "hold"
    TURN = "turn"
    FORWARD = "forward"
    APPROACH_FRONTIER = "approach_frontier"
    REVISIT_ANCHOR = "revisit_anchor"
    RECOVERY_SCAN = "recovery_scan"


@dataclass(frozen=True)
class PlannerCommand:
    mode: PlannerMode
    target_heading: float | None = None
    target_cell: tuple[int, int] | None = None
    reason: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class PlannerSnapshot:
    planner_mode: str
    frontier_count: int
    selected_frontier: tuple[float, float] | None
    coverage_ratio: float
    free_cells: int
    occupied_cells: int
    unknown_cells: int
    reason: str


class FrontierPlanner:
    def __init__(self) -> None:
        self._free_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint16)
        self._occ_hits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint16)
        self._selected_frontier_cell: tuple[int, int] | None = None
        self._selected_frontier_age = 0
        self._deferred_targets: dict[tuple[int, int], int] = {}
        self._tick = 0
        self._last_command = PlannerCommand(mode=PlannerMode.HOLD, reason="waiting_for_map")

    def defer_target(self, cell: tuple[int, int] | None) -> None:
        if cell is None:
            return
        self._deferred_targets[cell] = self._tick + DEFER_TARGET_STEPS
        if self._selected_frontier_cell == cell:
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0

    def update_from_frame(
        self,
        camera_pose: np.ndarray,
        planning_points: np.ndarray | None,
        *,
        pose_valid: bool,
        tracking_lost: bool,
    ) -> None:
        if not pose_valid or tracking_lost or planning_points is None or len(planning_points) == 0:
            return

        cam_x, cam_z = float(camera_pose[0, 3]), float(camera_pose[2, 3])
        cam_cell = self._world_to_cell(cam_x, cam_z)
        if cam_cell is None:
            return

        cam_y = float(camera_pose[1, 3])
        for pt in planning_points:
            if not np.isfinite(pt).all():
                continue
            px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
            if abs(py - cam_y) > 1.0:
                continue
            dist = math.hypot(px - cam_x, pz - cam_z)
            if dist < 0.08 or dist > MAX_OBSERVATION_RANGE_M:
                continue
            end_cell = self._world_to_cell(px, pz)
            if end_cell is None:
                continue
            ray = self._ray_cells(cam_cell, end_cell)
            if len(ray) < 2:
                continue
            for cell in ray[:-1]:
                self._free_hits[cell] = min(65535, int(self._free_hits[cell]) + 1)
            self._occ_hits[ray[-1]] = min(65535, int(self._occ_hits[ray[-1]]) + 1)

    def command_for_state(
        self,
        camera_pose: np.ndarray | None,
        *,
        pose_valid: bool,
        tracking_lost: bool,
        tracking_stable: bool,
    ) -> PlannerCommand:
        self._tick += 1
        expired = [cell for cell, until in self._deferred_targets.items() if until <= self._tick]
        for cell in expired:
            self._deferred_targets.pop(cell, None)

        if camera_pose is None or not pose_valid:
            self._last_command = PlannerCommand(
                mode=PlannerMode.HOLD,
                reason="pose_invalid",
                confidence=0.0,
            )
            return self._last_command

        if tracking_lost or not tracking_stable:
            heading = self._current_heading(camera_pose)
            self._last_command = PlannerCommand(
                mode=PlannerMode.REVISIT_ANCHOR,
                target_heading=heading,
                reason="tracking_unstable",
                confidence=TRACKING_UNSAFE_SCORE,
            )
            return self._last_command

        frontiers = self._frontier_clusters()
        if not frontiers:
            self._selected_frontier_cell = None
            self._selected_frontier_age = 0
            self._last_command = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=self._current_heading(camera_pose) + math.radians(25.0),
                reason="no_frontier_scan",
                confidence=0.2,
            )
            return self._last_command

        best_cell, best_score = self._select_frontier(camera_pose, frontiers)
        if self._selected_frontier_cell == best_cell:
            self._selected_frontier_age += 1
        else:
            self._selected_frontier_cell = best_cell
            self._selected_frontier_age = 1
        target_world = self._cell_to_world(*best_cell)
        dx = target_world[0] - float(camera_pose[0, 3])
        dz = target_world[1] - float(camera_pose[2, 3])
        target_heading = math.atan2(dz, dx)
        heading_error = self._wrap_angle(target_heading - self._current_heading(camera_pose))
        distance = math.hypot(dx, dz)

        if distance <= FRONTIER_REACHED_M:
            cmd = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=target_heading + math.radians(20.0),
                target_cell=best_cell,
                reason="frontier_reached_reorient",
                confidence=min(1.0, best_score),
            )
        elif abs(heading_error) > TURN_IN_PLACE_RAD:
            cmd = PlannerCommand(
                mode=PlannerMode.TURN,
                target_heading=target_heading,
                target_cell=best_cell,
                reason="align_frontier",
                confidence=min(1.0, best_score),
            )
        else:
            cmd = PlannerCommand(
                mode=PlannerMode.APPROACH_FRONTIER if abs(heading_error) < FORWARD_HEADING_RAD else PlannerMode.FORWARD,
                target_heading=target_heading,
                target_cell=best_cell,
                reason="approach_frontier",
                confidence=min(1.0, best_score),
            )

        self._last_command = cmd
        return cmd

    def snapshot(self) -> PlannerSnapshot:
        state = self._state_grid()
        free_cells = int(np.count_nonzero(state == 1))
        occupied_cells = int(np.count_nonzero(state == 2))
        unknown_cells = int(np.count_nonzero(state == 0))
        known = free_cells + occupied_cells
        total = known + unknown_cells
        coverage = (known / total) if total else 0.0
        selected = None
        if self._selected_frontier_cell is not None:
            selected = self._cell_to_world(*self._selected_frontier_cell)
        return PlannerSnapshot(
            planner_mode=self._last_command.mode.value,
            frontier_count=len(self._frontier_clusters()),
            selected_frontier=selected,
            coverage_ratio=coverage,
            free_cells=free_cells,
            occupied_cells=occupied_cells,
            unknown_cells=unknown_cells,
            reason=self._last_command.reason,
        )

    def _state_grid(self) -> np.ndarray:
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        free_mask = self._free_hits >= 2
        occ_mask = self._occ_hits >= 2
        state[free_mask] = 1
        state[occ_mask & (self._occ_hits >= self._free_hits)] = 2
        state[(state == 2) & (self._free_hits > self._occ_hits)] = 1
        return state

    def _frontier_clusters(self) -> list[list[tuple[int, int]]]:
        state = self._state_grid()
        free_mask = state == 1
        unknown_mask = state == 0
        frontier_mask = np.zeros_like(free_mask, dtype=bool)
        frontier_mask[1:-1, 1:-1] = free_mask[1:-1, 1:-1] & (
            unknown_mask[:-2, 1:-1] | unknown_mask[2:, 1:-1] | unknown_mask[1:-1, :-2] | unknown_mask[1:-1, 2:]
        )

        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: list[list[tuple[int, int]]] = []
        for row, col in np.argwhere(frontier_mask):
            row = int(row)
            col = int(col)
            if visited[row, col]:
                continue
            cluster: list[tuple[int, int]] = []
            queue = deque([(row, col)])
            visited[row, col] = True
            while queue:
                cur = queue.popleft()
                cluster.append(cur)
                for nxt in self._neighbors(*cur):
                    nr, nc = nxt
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and frontier_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append(nxt)
            if len(cluster) >= MIN_FRONTIER_CLUSTER:
                clusters.append(cluster)
        return clusters

    def _select_frontier(
        self,
        camera_pose: np.ndarray,
        clusters: list[list[tuple[int, int]]],
    ) -> tuple[tuple[int, int], float]:
        cur_x, cur_z = float(camera_pose[0, 3]), float(camera_pose[2, 3])
        heading = self._current_heading(camera_pose)
        best_cell = clusters[0][0]
        best_score = -1e9
        for cluster in clusters:
            rows = np.array([c[0] for c in cluster], dtype=np.float32)
            cols = np.array([c[1] for c in cluster], dtype=np.float32)
            center_cell = (int(round(rows.mean())), int(round(cols.mean())))
            wx, wz = self._cell_to_world(*center_cell)
            dist = math.hypot(wx - cur_x, wz - cur_z)
            target_heading = math.atan2(wz - cur_z, wx - cur_x)
            heading_cost = abs(self._wrap_angle(target_heading - heading))
            info_gain = len(cluster)
            score = (info_gain * 0.35) - (dist * 0.55) - (heading_cost * 0.90)
            if dist < MIN_TARGET_DISTANCE_M:
                score -= (MIN_TARGET_DISTANCE_M - dist) * 3.0
            if heading_cost < math.radians(35.0):
                score += 0.8
            if heading_cost > math.pi / 2:
                score -= REAR_TARGET_PENALTY * ((heading_cost - (math.pi / 2)) / (math.pi / 2))
            if center_cell in self._deferred_targets:
                score -= 10.0
            if self._selected_frontier_cell == center_cell:
                score += TARGET_SWITCH_BONUS
            if score > best_score:
                best_cell = center_cell
                best_score = score
        return best_cell, best_score

    def _neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        return [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

    def _world_to_cell(self, x: float, z: float) -> tuple[int, int] | None:
        col = int(round((x / GRID_RESOLUTION_M) + GRID_CENTER))
        row = int(round((z / GRID_RESOLUTION_M) + GRID_CENTER))
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None

    def _cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = (col - GRID_CENTER) * GRID_RESOLUTION_M
        z = (row - GRID_CENTER) * GRID_RESOLUTION_M
        return x, z

    def _ray_cells(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        sr, sc = start
        er, ec = end
        steps = int(min(MAX_RAY_SAMPLES, max(abs(er - sr), abs(ec - sc)) + 1))
        cells: list[tuple[int, int]] = []
        seen = set()
        for i in range(steps + 1):
            t = i / max(1, steps)
            row = int(round(sr + (er - sr) * t))
            col = int(round(sc + (ec - sc) * t))
            cell = (row, col)
            if cell in seen:
                continue
            seen.add(cell)
            cells.append(cell)
        return cells

    def _current_heading(self, camera_pose: np.ndarray) -> float:
        forward = camera_pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return math.atan2(float(forward[2]), float(forward[0]))

    def _wrap_angle(self, angle: float) -> float:
        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle
