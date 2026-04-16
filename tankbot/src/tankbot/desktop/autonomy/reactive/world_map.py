"""Sparse world-frame 2D occupancy map with visualization.

Accumulates obstacle points in world frame using scan-match pose
estimates. Used by both the live engine (for dashboard map image)
and the replay harness (for offline visualization).

The map is sparse (dict-backed) and unbounded — no fixed grid size.
Rendering centers on the robot's latest pose and shows a configurable
window around it.

Quality gating: the caller signals whether the current pose is
trustworthy. Only high-confidence frames contribute obstacle points
to the map. The pose trail is subsampled to avoid visual noise
from per-frame jitter.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class WorldMap:
    """Accumulates obstacle points and robot poses in world frame."""

    def __init__(
        self,
        *,
        resolution_m: float = 0.05,
        render_size_px: int = 480,
        trail_subsample: int = 5,
        min_cell_hits: int = 3,
    ) -> None:
        self._res = resolution_m
        self._render_size = render_size_px
        self._trail_subsample = trail_subsample
        self._min_cell_hits = min_cell_hits
        # Sparse world-frame occupancy counts: (cell_x, cell_y) -> hit count.
        self._cells: dict[tuple[int, int], int] = {}
        # Subsampled robot pose trail for rendering.
        self._trail: list[tuple[float, float]] = []
        # Latest pose (always tracked, for centering).
        self._latest_pose: tuple[float, float, float] | None = None
        self._frame_idx: int = 0

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    def add_frame(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        robot_points: np.ndarray,
        *,
        good_pose: bool = True,
    ) -> None:
        """Add a frame of robot-frame obstacle points, transformed to world.

        Parameters
        ----------
        robot_x, robot_y, robot_yaw
            Robot pose in world frame.
        robot_points
            (N, 3) robot-frame points (x forward, y left, z up).
        good_pose
            If False, only the trail is updated — obstacle points are
            NOT accumulated into the map. Use this to gate on scan-match
            quality / health.
        """
        self._latest_pose = (robot_x, robot_y, robot_yaw)
        self._frame_idx += 1

        # Subsample trail to reduce spaghetti.
        if self._frame_idx % self._trail_subsample == 0:
            self._trail.append((robot_x, robot_y))

        if not good_pose or robot_points.shape[0] == 0:
            return

        # Transform robot-frame points to world frame (2D).
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        px = robot_points[:, 0]
        py = robot_points[:, 1]
        wx = robot_x + cos_y * px - sin_y * py
        wy = robot_y + sin_y * px + cos_y * py

        cx = np.floor(wx / self._res).astype(np.int32)
        cy = np.floor(wy / self._res).astype(np.int32)

        for i in range(cx.shape[0]):
            key = (int(cx[i]), int(cy[i]))
            self._cells[key] = self._cells.get(key, 0) + 1

    def render(self, size: int | None = None) -> np.ndarray:
        """Render the map as a BGR image, centered on the latest pose."""
        sz = size or self._render_size
        img = np.full((sz, sz, 3), 30, dtype=np.uint8)

        if self._latest_pose is None:
            return img

        cx, cy, yaw = self._latest_pose

        # Draw occupied cells (only those with enough hits to be reliable).
        if self._cells:
            cells_arr = np.array(list(self._cells.keys()), dtype=np.float32)
            counts = np.array(list(self._cells.values()), dtype=np.float32)

            # Filter by minimum hit count to suppress noise.
            strong = counts >= self._min_cell_hits
            if strong.any():
                cells_arr = cells_arr[strong]
                counts = counts[strong]

                cell_wx = (cells_arr[:, 0] + 0.5) * self._res
                cell_wy = (cells_arr[:, 1] + 0.5) * self._res

                px_col = sz // 2 - ((cell_wy - cy) / self._res).astype(np.int32)
                px_row = sz // 2 - ((cell_wx - cx) / self._res).astype(np.int32)

                visible = (px_col >= 0) & (px_col < sz) & (px_row >= 0) & (px_row < sz)
                px_col = px_col[visible]
                px_row = px_row[visible]
                vis_counts = counts[visible]

                intensity = np.clip(np.log1p(vis_counts) * 30, 80, 255).astype(np.uint8)
                for r, c, v in zip(px_row, px_col, intensity, strict=True):
                    img[r, c] = (int(v), int(v), int(v))

        # Pose trail (green line, subsampled).
        if len(self._trail) >= 2:
            trail_px = []
            for tx, ty in self._trail:
                col = sz // 2 - int((ty - cy) / self._res)
                row = sz // 2 - int((tx - cx) / self._res)
                trail_px.append((col, row))
            for i in range(1, len(trail_px)):
                cv2.line(img, trail_px[i - 1], trail_px[i], (0, 160, 0), 1)

        # Current robot (bright green circle + heading arrow).
        center = sz // 2
        cv2.circle(img, (center, center), 4, (0, 255, 0), -1)
        arrow_len = 15
        ax = int(center - arrow_len * math.sin(yaw))
        ay = int(center - arrow_len * math.cos(yaw))
        cv2.arrowedLine(img, (center, center), (ax, ay), (0, 255, 0), 2)

        return img

    def clear(self) -> None:
        """Reset the map."""
        self._cells.clear()
        self._trail.clear()
        self._latest_pose = None
        self._frame_idx = 0
