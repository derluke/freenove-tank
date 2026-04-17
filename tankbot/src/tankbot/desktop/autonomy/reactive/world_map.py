"""Sparse world-frame 2D occupancy map with visualization.

Accumulates obstacle points in world frame using scan-match pose
estimates. Used by both the live engine (for dashboard map image)
and the replay harness (for offline visualization).

The map is sparse (dict-backed) and unbounded — no fixed grid size.
Rendering auto-zooms to the bounding box of all observed data, with
padding, so the map is always readable regardless of scale.

Quality gating: the caller signals whether the current pose is
trustworthy. Only high-confidence frames contribute obstacle points
to the map. The pose trail is subsampled to avoid visual noise
from per-frame jitter.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Minimum visible world extent (meters) — avoids over-zooming when the
# robot has barely moved.
_MIN_EXTENT_M = 2.0
# Padding added around the data bounding box (meters).
_BBOX_PAD_M = 0.5


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
        self._last_ppm: float = 0.0

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    @property
    def last_ppm(self) -> float:
        """Pixels per meter from the most recent render() call."""
        return self._last_ppm

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
        """Render the map as a BGR image, auto-zoomed to fit the data."""
        sz = size or self._render_size
        img = np.full((sz, sz, 3), 30, dtype=np.uint8)

        if self._latest_pose is None:
            return img

        robot_x, robot_y, yaw = self._latest_pose

        # Compute data bounding box in world coordinates.
        all_x = [robot_x]
        all_y = [robot_y]
        for tx, ty in self._trail:
            all_x.append(tx)
            all_y.append(ty)

        strong_cells = None
        strong_wx = None
        strong_wy = None
        if self._cells:
            cells_arr = np.array(list(self._cells.keys()), dtype=np.float32)
            counts = np.array(list(self._cells.values()), dtype=np.float32)
            strong = counts >= self._min_cell_hits
            if strong.any():
                strong_cells = cells_arr[strong]
                strong_wx = (strong_cells[:, 0] + 0.5) * self._res
                strong_wy = (strong_cells[:, 1] + 0.5) * self._res
                all_x.extend([float(strong_wx.min()), float(strong_wx.max())])
                all_y.extend([float(strong_wy.min()), float(strong_wy.max())])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Ensure a minimum visible extent so we don't over-zoom.
        extent_x = max(max_x - min_x + 2 * _BBOX_PAD_M, _MIN_EXTENT_M)
        extent_y = max(max_y - min_y + 2 * _BBOX_PAD_M, _MIN_EXTENT_M)
        extent = max(extent_x, extent_y)

        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5

        # Pixels per meter — scale the data to fill the render image.
        ppm = (sz - 4) / extent  # 2px margin on each side
        self._last_ppm = ppm

        def world_to_px(wx: float, wy: float) -> tuple[int, int]:
            col = int(sz / 2 + (wy - center_y) * ppm)
            row = int(sz / 2 - (wx - center_x) * ppm)
            return col, row

        # Draw occupied cells as visible filled squares.
        cell_px = max(2, int(self._res * ppm * 0.8))
        half_cell = cell_px // 2
        if strong_wx is not None and strong_wy is not None:
            counts_strong = counts[counts >= self._min_cell_hits]
            intensity = np.clip(np.log1p(counts_strong) * 30, 80, 255).astype(np.uint8)
            for i in range(len(strong_wx)):
                c, r = world_to_px(float(strong_wx[i]), float(strong_wy[i]))
                if 0 <= r < sz and 0 <= c < sz:
                    v = int(intensity[i])
                    r0 = max(0, r - half_cell)
                    r1 = min(sz, r + half_cell + 1)
                    c0 = max(0, c - half_cell)
                    c1 = min(sz, c + half_cell + 1)
                    img[r0:r1, c0:c1] = (v, v, v)

        # Pose trail (green line, subsampled).
        if len(self._trail) >= 2:
            trail_px = [world_to_px(tx, ty) for tx, ty in self._trail]
            line_w = max(1, int(ppm * 0.02))
            for i in range(1, len(trail_px)):
                cv2.line(img, trail_px[i - 1], trail_px[i], (0, 160, 0), line_w, cv2.LINE_AA)

        # Current robot (bright green circle + heading arrow).
        rc, rr = world_to_px(robot_x, robot_y)
        radius = max(4, int(ppm * 0.06))
        cv2.circle(img, (rc, rr), radius, (0, 255, 0), -1)
        arrow_len = max(12, int(ppm * 0.15))
        ax = int(rc - arrow_len * math.sin(yaw))
        ay = int(rr - arrow_len * math.cos(yaw))
        cv2.arrowedLine(img, (rc, rr), (ax, ay), (0, 255, 0), max(2, int(ppm * 0.02)), tipLength=0.3)

        return img

    def clear(self) -> None:
        """Reset the map."""
        self._cells.clear()
        self._trail.clear()
        self._latest_pose = None
        self._frame_idx = 0
