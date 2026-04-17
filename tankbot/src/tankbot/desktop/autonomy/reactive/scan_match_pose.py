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

IMU integration (Phase 2b):
When `gyro_yaw` is provided, the gyro is the **authoritative** rotation
source. The scan-match ICP receives a strong rotation prior so it only
solves for translation. The accumulated yaw comes directly from the
gyro integrator, not from the scan match. This eliminates the wild
heading jumps that monocular-depth scan matching produces.

Motor gating:
When the caller signals `motors_active=False`, the pose is frozen —
scan matching still runs for keyframe bookkeeping, but the published
pose does not change. This eliminates phantom drift from scan-match
noise when the robot is stationary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tankbot.desktop.autonomy.reactive.gyro import unwrap_delta
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
    """Mean residual below this = HEALTHY match (meters)."""

    residual_degraded: float = 0.08
    """Mean residual above this = degraded match."""

    min_correspondences: int = 20
    """Below this, the match is considered unreliable."""

    degraded_to_broken_frames: int = 10
    """Consecutive degraded/failed frames before BROKEN."""

    broken_to_healthy_frames: int = 5
    """Consecutive healthy frames to recover from BROKEN."""

    max_delta_m: float = 0.20
    """Reject ICP translation deltas larger than this since keyframe."""

    max_delta_yaw: float = 0.35
    """Reject ICP rotation deltas larger than this since keyframe
    (radians, ~20 deg). Only used when gyro is unavailable."""

    keyframe_dist_m: float = 0.15
    """Promote keyframe when translation exceeds this (meters)."""

    keyframe_yaw_rad: float = 0.20
    """Promote keyframe when rotation exceeds this (radians, ~11 deg)."""

    keyframe_max_frames: int = 60
    """Force keyframe promotion after this many frames."""

    xy_var_healthy: float = 0.01
    xy_var_degraded: float = 0.10
    yaw_var_base: float = 0.001

    pose_ema_alpha: float = 0.15
    """EMA smoothing for published pose. 0.15 at 23 Hz gives ~280 ms
    effective time constant — trades a bit of responsiveness for much
    smoother map rendering."""


class ScanMatchPoseSource(PoseSource):
    """Pose source using keyframe-based depth scan matching.

    When gyro_yaw is provided, the gyro is authoritative for rotation
    and scan-match only contributes translation. When gyro is absent,
    scan-match provides both rotation and translation (noisier).
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

        # Accumulated world-frame pose.
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
        self._keyframe_gyro_yaw: float | None = None
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

        self._last_match: MatchResult | None = None

    def latest(self) -> PoseEstimate:
        return self._latest

    @property
    def last_match(self) -> MatchResult | None:
        return self._last_match

    def update(
        self,
        points_xyz: np.ndarray,
        *,
        rotation_prior: float | None = None,
        gyro_yaw: float | None = None,
        motors_active: bool = True,
    ) -> MatchResult | None:
        """Feed a new frame of robot-frame 3D points.

        Parameters
        ----------
        points_xyz
            (N, 3) robot-frame points from depth projection.
        rotation_prior
            Pre-computed rotation-since-keyframe prior (radians).
            Most callers should pass `gyro_yaw` instead.
        gyro_yaw
            Absolute integrated gyro yaw (radians, CCW positive). When
            provided, the gyro is the **authoritative** rotation source.
        motors_active
            False when motors are stopped. Pose is frozen to prevent
            scan-match noise from drifting the position at rest.
        """
        now = self._clock()

        scan = extract_scan(points_xyz, self._scan_cfg)

        if self._keyframe_scan is None:
            self._keyframe_scan = scan
            self._keyframe_gyro_yaw = gyro_yaw
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

        # Derive rotation prior from absolute gyro yaw.
        gyro_delta: float | None = None
        if gyro_yaw is not None and self._keyframe_gyro_yaw is not None:
            gyro_delta = unwrap_delta(gyro_yaw - self._keyframe_gyro_yaw)
        if rotation_prior is None:
            rotation_prior = gyro_delta

        # Match current scan against keyframe.
        result = match_scans(
            source=scan,
            reference=self._keyframe_scan,
            config=self._icp_cfg,
            rotation_prior=rotation_prior,
        )
        self._last_match = result

        # --- Rotation: gyro is authoritative, applied unconditionally ---
        # This is independent of scan-match quality. Even when ICP fails
        # (which is common with monocular depth), the heading tracks reality.
        # Gyro yaw bypasses EMA — the hardware DLPF already smooths it.
        gyro_authoritative = False
        if gyro_delta is not None and motors_active:
            self._yaw = self._keyframe_yaw + gyro_delta
            self._yaw_smooth = self._yaw
            gyro_authoritative = True

        # --- Translation: only from good scan-match when motors are active ---
        if not motors_active:
            if self._frames_since_keyframe >= self._cfg.keyframe_max_frames:
                self._promote_keyframe(scan, gyro_yaw)
            self._update_smoothed(now)
            return result

        good = self._evaluate_match(result)

        if good:
            delta_m = math.hypot(result.dx, result.dy)
            yaw_too_big = gyro_delta is None and abs(result.dyaw) > self._cfg.max_delta_yaw
            if delta_m > self._cfg.max_delta_m or yaw_too_big:
                self._mark_degraded()
            else:
                cos_k = math.cos(self._keyframe_yaw)
                sin_k = math.sin(self._keyframe_yaw)
                self._x = self._keyframe_x + cos_k * result.dx - sin_k * result.dy
                self._y = self._keyframe_y + sin_k * result.dx + cos_k * result.dy

                # If no gyro, also take rotation from scan-match.
                if gyro_delta is None:
                    self._yaw = self._keyframe_yaw + result.dyaw

                self._mark_healthy()
                self._maybe_promote_keyframe(scan, delta_m, abs(gyro_delta or result.dyaw), gyro_yaw)
        else:
            self._mark_degraded()
            # Promote keyframe on gyro rotation even when ICP fails,
            # so the reference scan stays aligned with reality.
            yaw_since_kf = abs(gyro_delta) if gyro_delta is not None else 0.0
            if (
                self._frames_since_keyframe > self._cfg.keyframe_max_frames
                or yaw_since_kf > self._cfg.keyframe_yaw_rad
            ):
                self._promote_keyframe(scan, gyro_yaw)

        self._update_smoothed(now, skip_yaw_ema=gyro_authoritative)
        return result

    def _update_smoothed(self, now: float, *, skip_yaw_ema: bool = False) -> None:
        """Apply EMA smoothing and publish the pose estimate."""
        a = self._cfg.pose_ema_alpha
        self._x_smooth += a * (self._x - self._x_smooth)
        self._y_smooth += a * (self._y - self._y_smooth)
        if not skip_yaw_ema:
            self._yaw_smooth += a * (self._yaw - self._yaw_smooth)

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

    def _maybe_promote_keyframe(
        self, scan: Scan2D, delta_m: float, delta_yaw: float, gyro_yaw: float | None
    ) -> None:
        if (
            delta_m >= self._cfg.keyframe_dist_m
            or delta_yaw >= self._cfg.keyframe_yaw_rad
            or self._frames_since_keyframe >= self._cfg.keyframe_max_frames
        ):
            self._promote_keyframe(scan, gyro_yaw)

    def _promote_keyframe(self, scan: Scan2D, gyro_yaw: float | None) -> None:
        self._keyframe_scan = scan
        self._keyframe_x = self._x
        self._keyframe_y = self._y
        self._keyframe_yaw = self._yaw
        self._keyframe_gyro_yaw = gyro_yaw
        self._frames_since_keyframe = 0

    def _evaluate_match(self, result: MatchResult) -> bool:
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

    def _mark_degraded(self) -> None:
        self._consecutive_healthy = 0
        self._consecutive_degraded += 1
        if self._health == HealthState.HEALTHY:
            self._health = HealthState.DEGRADED
        if self._consecutive_degraded >= self._cfg.degraded_to_broken_frames:
            self._health = HealthState.BROKEN

    def reset(self) -> None:
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
        self._keyframe_gyro_yaw = None
        self._frames_since_keyframe = 0
        self._last_match = None
