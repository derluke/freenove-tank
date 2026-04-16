"""Scan-match pose source — Phase 2 translation estimator.

Per the Phase 2 plan: depth-based 2D scan matching as a `PoseSource`
implementation. Projects depth into a synthetic 2D laser scan, ICPs
against a **keyframe** reference scan, accumulates deltas into a
world-frame pose.

Keyframe strategy: matching consecutive frames at 23 Hz produces
sub-centimeter real displacement that is swamped by depth-model noise.
Instead we hold a reference keyframe and match every incoming scan
against it. When the accumulated displacement since the keyframe
exceeds `keyframe_dist_m` (or rotation exceeds `keyframe_yaw_rad`),
the current scan becomes the new keyframe. This gives ICP a larger,
resolvable displacement to work with.

Health transitions:
- HEALTHY: scan match converges with mean residual below threshold.
- DEGRADED: residual elevated or correspondence count low for a few
  ticks. `xy_m` still reported but `xy_var` inflated.
- BROKEN: N consecutive degraded ticks. `xy_m` reverts to None.
  Recovery requires N_healthy consecutive good matches.

Without IMU (Phase 2 initial): rotation comes from the scan match
itself. With IMU (future): gyro provides a rotation prior, scan match
refines translation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tankbot.desktop.autonomy.reactive.pose import (
    HealthState,
    PoseEstimate,
    PoseSource,
)
from tankbot.desktop.autonomy.reactive.scan import Scan2D, ScanConfig, extract_scan
from tankbot.desktop.autonomy.reactive.scan_match import (
    ICPConfig,
    MatchResult,
    match_scans,
)


@dataclass
class ScanMatchConfig:
    """Tunables for the scan-match pose source."""

    residual_healthy: float = 0.03
    """Mean residual below this = HEALTHY match (meters). Tight
    threshold — only high-confidence matches update the pose."""

    residual_degraded: float = 0.08
    """Mean residual above this = degraded match. Between healthy and
    degraded is a gray zone that counts toward degradation."""

    min_correspondences: int = 20
    """Below this, the match is considered unreliable regardless of
    residual."""

    degraded_to_broken_frames: int = 10
    """Consecutive degraded/failed frames before transitioning to BROKEN."""

    broken_to_healthy_frames: int = 5
    """Consecutive healthy frames required to recover from BROKEN."""

    max_delta_m: float = 0.30
    """Reject scan-match deltas larger than this since the keyframe."""

    max_delta_yaw: float = 0.50
    """Reject scan-match rotation deltas larger than this since the
    keyframe (radians, ~28 deg)."""

    keyframe_dist_m: float = 0.15
    """Promote current scan to keyframe when accumulated translation
    since the last keyframe exceeds this (meters). Larger = fewer
    keyframe switches = less noise accumulation."""

    keyframe_yaw_rad: float = 0.20
    """Promote current scan to keyframe when accumulated rotation
    since the last keyframe exceeds this (radians, ~11 deg)."""

    keyframe_max_frames: int = 60
    """Force keyframe promotion after this many frames even if the
    displacement threshold hasn't been reached. Prevents the reference
    from going stale."""

    xy_var_healthy: float = 0.01
    """Reported xy variance when healthy (m^2)."""

    xy_var_degraded: float = 0.10
    """Reported xy variance when degraded (m^2)."""

    yaw_var_base: float = 0.001
    """Base yaw variance (rad^2)."""

    pose_ema_alpha: float = 0.25
    """EMA smoothing factor for the published pose. Lower = smoother
    but slower to track real motion. 0.25 at 23 Hz gives ~170 ms
    effective time constant — enough to kill per-frame jitter while
    tracking cruise-speed motion."""


class ScanMatchPoseSource(PoseSource):
    """Pose source using keyframe-based depth scan matching.

    Feed it robot-frame 3D points each tick via `update()`. It extracts
    a synthetic 2D scan, matches against a keyframe reference scan, and
    accumulates the pose. The keyframe is promoted when accumulated
    displacement exceeds the configured thresholds.
    """

    source_name = "scan_match"

    def __init__(
        self,
        *,
        scan_config: ScanConfig | None = None,
        icp_config: ICPConfig | None = None,
        config: ScanMatchConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scan_cfg = scan_config or ScanConfig()
        self._icp_cfg = icp_config or ICPConfig()
        self._cfg = config or ScanMatchConfig()
        self._clock = clock

        # Accumulated world-frame pose (raw, from ICP).
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0

        # EMA-smoothed pose (published via latest()).
        self._x_smooth: float = 0.0
        self._y_smooth: float = 0.0
        self._yaw_smooth: float = 0.0

        # Health tracking.
        self._health = HealthState.HEALTHY
        self._consecutive_degraded: int = 0
        self._consecutive_healthy: int = 0

        # Keyframe state.
        self._keyframe_scan: Scan2D | None = None
        self._keyframe_x: float = 0.0
        self._keyframe_y: float = 0.0
        self._keyframe_yaw: float = 0.0
        self._frames_since_keyframe: int = 0

        # Latest estimate (returned by latest()).
        self._latest = PoseEstimate(
            t_monotonic=clock(),
            yaw_rad=0.0,
            yaw_var=0.0,
            xy_m=(0.0, 0.0),
            xy_var=self._cfg.xy_var_healthy,
            health=HealthState.HEALTHY,
            source=self.source_name,
        )

        # Last match result for debugging/replay.
        self._last_match: MatchResult | None = None

    def latest(self) -> PoseEstimate:
        return self._latest

    @property
    def last_match(self) -> MatchResult | None:
        """Most recent ICP match result, for debugging/visualization."""
        return self._last_match

    def update(
        self,
        points_xyz: np.ndarray,
        *,
        rotation_prior: float | None = None,
    ) -> MatchResult | None:
        """Feed a new frame of robot-frame 3D points.

        Parameters
        ----------
        points_xyz
            (N, 3) robot-frame points from depth projection.
        rotation_prior
            Optional gyro-derived rotation since last keyframe (radians).
            When IMU is available, pass the integrated gyro delta here.

        Returns
        -------
        MatchResult or None
            The ICP result, or None if this was the first frame.
        """
        now = self._clock()

        # Extract synthetic scan.
        scan = extract_scan(points_xyz, self._scan_cfg)

        if self._keyframe_scan is None:
            # First frame — set as keyframe.
            self._keyframe_scan = scan
            self._latest = PoseEstimate(
                t_monotonic=now,
                yaw_rad=0.0,
                yaw_var=self._cfg.yaw_var_base,
                xy_m=(0.0, 0.0),
                xy_var=self._cfg.xy_var_healthy,
                health=HealthState.HEALTHY,
                source=self.source_name,
            )
            return None

        self._frames_since_keyframe += 1

        # Match current scan against keyframe.
        result = match_scans(
            source=scan,
            reference=self._keyframe_scan,
            config=self._icp_cfg,
            rotation_prior=rotation_prior,
        )
        self._last_match = result

        # Evaluate match quality.
        good = self._evaluate_match(result)

        if good:
            # Sanity check on delta magnitude.
            delta_m = math.hypot(result.dx, result.dy)
            if delta_m > self._cfg.max_delta_m or abs(result.dyaw) > self._cfg.max_delta_yaw:
                self._mark_degraded()
            else:
                # The ICP result is the transform from keyframe to current.
                # Compute new absolute pose = keyframe pose + delta.
                cos_k = math.cos(self._keyframe_yaw)
                sin_k = math.sin(self._keyframe_yaw)
                self._x = self._keyframe_x + cos_k * result.dx - sin_k * result.dy
                self._y = self._keyframe_y + sin_k * result.dx + cos_k * result.dy
                self._yaw = self._keyframe_yaw + result.dyaw
                self._mark_healthy()

                # Check if we should promote to a new keyframe.
                self._maybe_promote_keyframe(scan, delta_m, abs(result.dyaw))
        else:
            self._mark_degraded()
            # If the match failed and we're far from the keyframe,
            # force a keyframe reset to avoid accumulating stale error.
            if self._frames_since_keyframe > self._cfg.keyframe_max_frames:
                self._promote_keyframe(scan)

        # EMA-smooth the raw pose to kill per-frame jitter.
        a = self._cfg.pose_ema_alpha
        self._x_smooth += a * (self._x - self._x_smooth)
        self._y_smooth += a * (self._y - self._y_smooth)
        self._yaw_smooth += a * (self._yaw - self._yaw_smooth)

        # Build the pose estimate from smoothed values.
        if self._health == HealthState.BROKEN:
            xy_m = None
            xy_var = None
        elif self._health == HealthState.DEGRADED:
            xy_m = (self._x_smooth, self._y_smooth)
            xy_var = self._cfg.xy_var_degraded
        else:
            xy_m = (self._x_smooth, self._y_smooth)
            xy_var = self._cfg.xy_var_healthy

        self._latest = PoseEstimate(
            t_monotonic=now,
            yaw_rad=self._yaw_smooth,
            yaw_var=self._cfg.yaw_var_base,
            xy_m=xy_m,
            xy_var=xy_var,
            health=self._health,
            source=self.source_name,
        )
        return result

    def _maybe_promote_keyframe(self, scan: Scan2D, delta_m: float, delta_yaw: float) -> None:
        """Promote current scan to keyframe if displacement is large enough."""
        if (
            delta_m >= self._cfg.keyframe_dist_m
            or delta_yaw >= self._cfg.keyframe_yaw_rad
            or self._frames_since_keyframe >= self._cfg.keyframe_max_frames
        ):
            self._promote_keyframe(scan)

    def _promote_keyframe(self, scan: Scan2D) -> None:
        """Set the current scan as the new keyframe reference."""
        self._keyframe_scan = scan
        self._keyframe_x = self._x
        self._keyframe_y = self._y
        self._keyframe_yaw = self._yaw
        self._frames_since_keyframe = 0

    def _evaluate_match(self, result: MatchResult) -> bool:
        """Return True if the match is good enough to use."""
        if not result.converged:
            return False
        if result.n_correspondences < self._cfg.min_correspondences:
            return False
        return result.mean_residual <= self._cfg.residual_degraded

    def _mark_healthy(self) -> None:
        self._consecutive_degraded = 0
        self._consecutive_healthy += 1
        if self._health == HealthState.BROKEN:
            if self._consecutive_healthy >= self._cfg.broken_to_healthy_frames:
                self._health = HealthState.HEALTHY
                self._consecutive_healthy = 0
        elif self._health == HealthState.DEGRADED:
            self._health = HealthState.HEALTHY
        # Already HEALTHY — no-op.

    def _mark_degraded(self) -> None:
        self._consecutive_healthy = 0
        self._consecutive_degraded += 1
        if self._health == HealthState.HEALTHY:
            self._health = HealthState.DEGRADED
        if self._consecutive_degraded >= self._cfg.degraded_to_broken_frames:
            self._health = HealthState.BROKEN

    def reset(self) -> None:
        """Reset accumulated pose and health state."""
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._x_smooth = 0.0
        self._y_smooth = 0.0
        self._yaw_smooth = 0.0
        self._health = HealthState.HEALTHY
        self._consecutive_degraded = 0
        self._consecutive_healthy = 0
        self._keyframe_scan = None
        self._keyframe_x = 0.0
        self._keyframe_y = 0.0
        self._keyframe_yaw = 0.0
        self._frames_since_keyframe = 0
        self._last_match = None
