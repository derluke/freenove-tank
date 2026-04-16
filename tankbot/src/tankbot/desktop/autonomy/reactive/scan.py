"""Synthetic 2D laser scan from depth-projected robot-frame points.

Per the Phase 2 plan: "project depth into a synthetic 2D laser scan
(nearest obstacle per angular bin from a floor-parallel slice)."

The scan is a fixed-size array of ranges at uniform angular spacing,
mimicking a planar lidar. Invalid bins (no points fell in them) are
set to `max_range_m`. This representation feeds directly into ICP
scan matching.

Conventions:
- Angles are measured from +x (forward) in the robot frame,
  counter-clockwise positive (standard math convention, consistent
  with the robot frame where +y = left).
- The scan covers a configurable angular range (default: 180 deg
  frontal arc, -90 to +90 deg) since the camera only sees forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanConfig:
    """Parameters for synthetic 2D scan extraction."""

    n_bins: int = 180
    """Number of angular bins. 180 bins over 180 deg = 1 deg resolution."""

    min_angle_rad: float = -np.pi / 2
    """Start angle (rad). Default -pi/2 (right)."""

    max_angle_rad: float = np.pi / 2
    """End angle (rad). Default +pi/2 (left)."""

    max_range_m: float = 5.0
    """Bins with no points are set to this value."""

    min_range_m: float = 0.10
    """Drop points closer than this (lens glare, self-hits)."""

    min_height_m: float = 0.02
    """Floor-parallel slice lower bound (robot frame z)."""

    max_height_m: float = 0.35
    """Floor-parallel slice upper bound (robot frame z).
    Only points in this band contribute to the scan — this
    filters out floor and ceiling returns."""


@dataclass(frozen=True)
class Scan2D:
    """A synthetic 2D laser scan."""

    ranges: np.ndarray
    """(n_bins,) float32 array of range values in meters.
    Invalid bins contain `max_range_m`."""

    angles: np.ndarray
    """(n_bins,) float32 array of bin center angles in radians."""

    config: ScanConfig
    """The config that produced this scan."""

    n_valid_bins: int
    """Number of bins that received at least one point."""

    def to_cartesian(self) -> np.ndarray:
        """Convert to (N, 2) cartesian points, excluding max-range bins."""
        valid = self.ranges < self.config.max_range_m
        r = self.ranges[valid]
        a = self.angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        return np.stack([x, y], axis=-1).astype(np.float32)


def extract_scan(
    points_xyz: np.ndarray,
    config: ScanConfig | None = None,
) -> Scan2D:
    """Extract a synthetic 2D scan from robot-frame 3D points.

    Parameters
    ----------
    points_xyz
        (N, 3) float array of robot-frame points (x forward, y left, z up).
        Typically the output of `project_depth_to_robot_points`.
    config
        Scan extraction parameters. Uses defaults if None.

    Returns
    -------
    Scan2D
        The synthetic scan.
    """
    cfg = config or ScanConfig()
    n = cfg.n_bins

    # Bin edges and centers.
    bin_edges = np.linspace(cfg.min_angle_rad, cfg.max_angle_rad, n + 1, dtype=np.float32)
    angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Start with all bins at max range.
    ranges = np.full(n, cfg.max_range_m, dtype=np.float32)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=0)

    if points_xyz.shape[0] == 0:
        return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=0)

    # Floor-parallel slice: filter by height.
    z = points_xyz[:, 2]
    height_mask = (z >= cfg.min_height_m) & (z <= cfg.max_height_m)
    pts = points_xyz[height_mask]

    if pts.shape[0] == 0:
        return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=0)

    # Project to 2D range and bearing.
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.sqrt(x * x + y * y)
    bearing = np.arctan2(y, x)

    # Filter by range.
    range_mask = (r >= cfg.min_range_m) & (r <= cfg.max_range_m)
    r = r[range_mask]
    bearing = bearing[range_mask]

    if r.shape[0] == 0:
        return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=0)

    # Digitize into angular bins. np.digitize returns 1-based indices
    # for values in [edges[i-1], edges[i]).
    bin_idx = np.digitize(bearing, bin_edges) - 1
    # Clamp to valid range (points exactly at max_angle land in bin n).
    valid_bin = (bin_idx >= 0) & (bin_idx < n)
    r = r[valid_bin]
    bin_idx = bin_idx[valid_bin]

    if r.shape[0] == 0:
        return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=0)

    # Take the minimum range per bin (nearest obstacle).
    # np.minimum.at does unbuffered in-place reduction.
    np.minimum.at(ranges, bin_idx, r.astype(np.float32))

    n_valid = int(np.sum(ranges < cfg.max_range_m))
    return Scan2D(ranges=ranges, angles=angles, config=cfg, n_valid_bins=n_valid)
