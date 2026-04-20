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
from collections import deque
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

    residual_healthy: float = 0.10
    """Residual below this = HEALTHY match. Relaxed for the common-bin
    forward estimator under monocular depth; useful matches regularly
    land above the old lidar-era threshold."""

    residual_degraded: float = 0.25
    """Residual above this = degraded match. The gyro-guided common-bin
    estimator sits around 0.02-0.23 m MAD on the Phase 0c set; 0.25 m
    keeps straight runs online while still rejecting obvious mismatch."""

    min_correspondences: int = 10
    """Below this, the match is considered unreliable. Lowered from 20
    because typical scan point counts drop to ~20-30 when the robot is
    close to a wall or on a sparse scene."""

    degraded_to_broken_frames: int = 20
    """Consecutive degraded/failed frames before BROKEN. Raised from 10
    because a single noisy keyframe promotion can cause a brief burst
    of degraded matches that recovers on its own."""

    broken_to_healthy_frames: int = 5
    """Consecutive healthy frames to recover from BROKEN."""

    max_delta_m: float = 0.20
    """Reject ICP translation deltas larger than this since keyframe."""

    common_bin_max_delta_m: float = 0.70
    """Reject gyro-guided common-bin translation deltas larger than this
    since keyframe. Monocular-depth common-bin estimates on straight
    runs regularly land above the old 0.20 m ICP cap, but turn-induced
    outliers are still meter-scale and should be blocked."""

    max_delta_yaw: float = 0.35
    """Reject ICP rotation deltas larger than this since keyframe
    (radians, ~20 deg). Only used when gyro is unavailable."""

    keyframe_dist_m: float = 0.15
    """Promote keyframe when translation exceeds this (meters)."""

    keyframe_yaw_rad: float = 0.15
    """Promote keyframe when rotation exceeds this (radians, ~11 deg)."""

    keyframe_max_frames: int = 40
    """Force keyframe promotion after this many frames."""

    motor_coast_hold_s: float = 0.35
    """Hold translation enabled briefly after motors drop to zero.
    The robot is pulse-driven, then coasts; the sampled motor channel
    captures the pulse more reliably than the coast. A short hold keeps
    translation alive across that gap without leaving the estimator on
    through long idle stretches."""

    xy_var_healthy: float = 0.01
    xy_var_degraded: float = 0.10
    yaw_var_base: float = 0.001

    pose_ema_alpha: float = 0.15
    """EMA smoothing for published pose. 0.15 at 23 Hz gives ~280 ms
    effective time constant — trades a bit of responsiveness for much
    smoother map rendering."""

    min_healthy_frames: int = 1
    """Consecutive good matches required to promote DEGRADED -> HEALTHY.
    §3.4 N_healthy. Default 1 preserves the pre-§3.4 "flap back on any
    good match" behavior because monocular-depth ICP produces inherently
    noisy per-frame deltas on straight drives — good/bad frames
    interleave, and requiring a run of N good frames in a row locks the
    state in DEGRADED forever. Raise when scan-match gets a stronger
    estimator (VIO) where runs of good matches are plausible."""

    max_rotation_rate_rad_s: float = 0.35
    """§3.4 τ_bias (rotation-gate). When |gyro rate| exceeds this AND
    motors are active, the xy commit is skipped (ICP fabricates phantom
    translation on monocular depth slices of a rotating wall — see
    Phase 0c turn_in_place_90). Health is **not** marked degraded by
    this rule: a brief swerve on a straight drive should not cascade
    to BROKEN. Only if rotation persists for ≥`rotation_degrade_frames`
    consecutive frames do we demote. ~20 deg/s."""

    rotation_degrade_frames: int = 10
    """Consecutive rotating-fast frames before marking DEGRADED.
    ~430 ms at 23 Hz. Longer than the typical swerve spike but shorter
    than an intentional in-place turn."""

    implied_speed_window_s: float = 0.25
    """§3.4 τ_trans window length. Candidate commits are compared against
    the most recent accepted pose at least this old, not against the
    immediately-prior frame. ~250 ms is long enough to average out the
    2-6 cm/frame monocular-depth jitter measured on Phase 0c while still
    short enough to catch genuine divergence quickly."""

    max_implied_speed_m_s: float = float("inf")
    """§3.4 τ_trans (velocity sanity gate). Disabled by default until a
    replay-tuned threshold/window pairing proves it improves motion
    estimation instead of suppressing valid forward progress."""

    implausible_run_to_broken: int = 0
    """0 disables escalation from velocity alone. The helper remains so
    replay tuning can re-enable it without API churn."""

    icp_smoothing_window: int = 1
    """Median-filter window for the no-gyro ICP fallback. Default 1:
    medianing absolute keyframe-relative deltas hid real forward motion
    on the straight replay set."""

    use_common_bin_translation_with_gyro: bool = True
    """When gyro yaw is available, estimate translation from same-angle
    scan bins instead of full 2D ICP. On wall-dominated scenes this is
    materially more stable for forward motion."""

    common_bin_max_angle_rad: float = 0.35
    """Use only a fairly narrow central field of view for gyro-guided
    translation. Wider bands help some scenes, but on the Phase 0c set
    they also pull in side structure that frequently flips the forward
    estimate sign on long straight runs."""

    common_bin_keep_nearest_n: int = 18
    """Within the central band, keep only the nearest bin pairs when
    estimating forward translation. Those bins are the best proxy for
    actual frontal motion; farther bins are more likely to be side
    structure whose apparent motion is dominated by monocular depth
    bias rather than robot translation."""

    common_bin_max_gyro_delta_rad: float = 0.12
    """Only use the common-bin translation path while yaw since the
    keyframe stays modest. Beyond ~9 deg, same-angle bins no longer
    refer to the same world patch under monocular depth, so translation
    is frozen and the keyframe is rotated forward instead."""


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

        # Velocity + rotation-rate tracking (§3.4 thresholds).
        self._last_gyro_yaw: float | None = None
        self._last_update_t: float | None = None
        self._consecutive_implausible: int = 0
        self._consecutive_rotating: int = 0
        self._commit_history: deque[tuple[float, float, float]] = deque()
        self._motion_allowed_until: float = float("-inf")

        # Rolling median filter on ICP (dx, dy) against the current keyframe.
        window = max(1, int(self._cfg.icp_smoothing_window))
        self._icp_dx_window: deque[float] = deque(maxlen=window)
        self._icp_dy_window: deque[float] = deque(maxlen=window)

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
        last_motor_active_t: float | None = None,
        t_monotonic: float | None = None,
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
        last_motor_active_t
            Timestamp of the most recent nonzero motor event. When
            provided, motor_coast_hold_s is measured from the actual
            event time rather than the sampled frame-time motor state.
        t_monotonic
            Observation timestamp. When provided, the pose source uses
            it as the authoritative "now" for rate computations. Omit
            in live operation (defaults to the injected clock). Replay
            tools should pass the observation timestamp so per-frame
            gyro-rate estimates align with data time rather than
            wall-clock time, which in replay can be arbitrarily
            different from the real inter-frame interval.
        """
        now = t_monotonic if t_monotonic is not None else self._clock()

        # Per-tick gyro rate (for §3.4 rotation-gate).
        gyro_rate_rad_s = 0.0
        if (
            gyro_yaw is not None
            and self._last_gyro_yaw is not None
            and self._last_update_t is not None
        ):
            dt = now - self._last_update_t
            if dt > 1e-6:
                gyro_rate_rad_s = unwrap_delta(gyro_yaw - self._last_gyro_yaw) / dt
        self._last_gyro_yaw = gyro_yaw
        self._last_update_t = now

        scan = extract_scan(points_xyz, self._scan_cfg)

        if self._keyframe_scan is None:
            self._keyframe_scan = scan
            self._keyframe_gyro_yaw = gyro_yaw
            self._record_commit(now)
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

        if last_motor_active_t is not None:
            self._motion_allowed_until = max(
                self._motion_allowed_until,
                last_motor_active_t + self._cfg.motor_coast_hold_s,
            )
        if motors_active:
            self._motion_allowed_until = max(
                self._motion_allowed_until,
                now + self._cfg.motor_coast_hold_s,
            )
        motion_allowed = motors_active or now <= self._motion_allowed_until

        # Match current scan against keyframe.
        if self._cfg.use_common_bin_translation_with_gyro and gyro_delta is not None:
            result = self._match_common_bin_translation(scan, gyro_delta)
        else:
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
        if gyro_delta is not None:
            self._yaw = self._keyframe_yaw + gyro_delta
            self._yaw_smooth = self._yaw
            gyro_authoritative = True

        # --- Translation: only from good scan-match when motors are active ---
        if not motion_allowed:
            # While translation is frozen, don't time-promote the
            # keyframe. If the robot coasted during a motor-off gap,
            # time-based promotion would silently discard that motion.
            if gyro_delta is not None and abs(gyro_delta) >= self._cfg.keyframe_yaw_rad:
                self._promote_keyframe(scan, gyro_yaw)
            self._update_smoothed(now)
            return result

        good = self._evaluate_match(result)

        rotating_fast = (
            gyro_yaw is not None
            and abs(gyro_rate_rad_s) > self._cfg.max_rotation_rate_rad_s
            and gyro_delta is not None
            and abs(gyro_delta) <= self._cfg.common_bin_max_gyro_delta_rad
        )

        if good:
            if gyro_delta is None:
                # Only smooth the no-gyro ICP fallback. The gyro-guided
                # common-bin estimator already uses a robust median, and
                # smoothing absolute keyframe-relative deltas hides real
                # motion.
                self._icp_dx_window.append(result.dx)
                self._icp_dy_window.append(result.dy)
                smoothed_dx = float(np.median(self._icp_dx_window))
                smoothed_dy = float(np.median(self._icp_dy_window))
            else:
                smoothed_dx = result.dx
                smoothed_dy = result.dy

            delta_m = math.hypot(smoothed_dx, smoothed_dy)
            common_bin_out_of_range = (
                gyro_delta is not None
                and self._cfg.use_common_bin_translation_with_gyro
                and abs(gyro_delta) > self._cfg.common_bin_max_gyro_delta_rad
            )
            max_translation_m = (
                self._cfg.common_bin_max_delta_m
                if gyro_delta is not None and self._cfg.use_common_bin_translation_with_gyro
                else self._cfg.max_delta_m
            )
            yaw_too_big = gyro_delta is None and abs(result.dyaw) > self._cfg.max_delta_yaw
            if common_bin_out_of_range:
                # Same-angle bins stop being meaningful once the robot has
                # rotated too far relative to the keyframe. Hold xy steady,
                # let gyro yaw continue, and promote the keyframe on
                # accumulated rotation so translation resumes on the next
                # straight-ish segment.
                self._consecutive_implausible = 0
                self._consecutive_rotating = 0
                self._maybe_promote_keyframe(scan, 0.0, abs(gyro_delta), gyro_yaw)
            elif delta_m > max_translation_m or yaw_too_big:
                self._mark_degraded()
            elif rotating_fast:
                # τ_bias: pure rotation makes ICP fabricate phantom xy
                # translation. Skip the xy commit silently — heading
                # already advanced via gyro above. Only demote to
                # DEGRADED once rotation has persisted long enough that
                # we're clearly in an in-place turn rather than a brief
                # swerve.
                self._consecutive_rotating += 1
                if self._consecutive_rotating >= self._cfg.rotation_degrade_frames:
                    self._mark_degraded()
                # Still promote keyframe on accumulated rotation so the
                # reference scan stays aligned.
                self._maybe_promote_keyframe(
                    scan, 0.0, abs(gyro_delta or result.dyaw), gyro_yaw,
                )
            else:
                cos_k = math.cos(self._keyframe_yaw)
                sin_k = math.sin(self._keyframe_yaw)
                new_x = self._keyframe_x + cos_k * smoothed_dx - sin_k * smoothed_dy
                new_y = self._keyframe_y + sin_k * smoothed_dx + cos_k * smoothed_dy

                if self._implausible_step(new_x, new_y, now):
                    # τ_trans: skip commits whose windowed speed is not
                    # physically plausible. The check runs on a
                    # 200-300 ms baseline so it rejects sustained
                    # divergence without false-positiving normal
                    # monocular-depth per-frame jitter.
                    self._consecutive_implausible += 1
                    if (
                        self._cfg.implausible_run_to_broken > 0
                        and self._consecutive_implausible >= self._cfg.implausible_run_to_broken
                    ):
                        self._mark_broken()
                else:
                    self._consecutive_implausible = 0
                    self._consecutive_rotating = 0
                    self._x = new_x
                    self._y = new_y

                    # If no gyro, also take rotation from scan-match.
                    if gyro_delta is None:
                        self._yaw = self._keyframe_yaw + result.dyaw

                    self._mark_healthy()
                    self._record_commit(now)
                    self._maybe_promote_keyframe(
                        scan, delta_m, abs(gyro_delta or result.dyaw), gyro_yaw,
                    )
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
        # ICP deltas are relative to the keyframe — they have no meaning
        # after promotion, so flush the filter window.
        self._icp_dx_window.clear()
        self._icp_dy_window.clear()

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
        elif (
            self._health == HealthState.DEGRADED
            and self._consecutive_healthy >= self._cfg.min_healthy_frames
        ):
            self._health = HealthState.HEALTHY

    def _mark_degraded(self) -> None:
        self._consecutive_healthy = 0
        self._consecutive_degraded += 1
        if self._health == HealthState.HEALTHY:
            self._health = HealthState.DEGRADED
        if self._consecutive_degraded >= self._cfg.degraded_to_broken_frames:
            self._health = HealthState.BROKEN

    def _mark_broken(self) -> None:
        """Hard-fail transition used when a sanity check trips.

        Unlike `_mark_degraded` this sets BROKEN directly rather than
        waiting for a run of degraded frames. Use it for implausible
        velocity, NaN poses, or other nonrecoverable-without-reset
        states per §3.4.
        """
        self._consecutive_healthy = 0
        self._consecutive_degraded = self._cfg.degraded_to_broken_frames
        self._health = HealthState.BROKEN

    def _implausible_step(self, new_x: float, new_y: float, now: float) -> bool:
        """True if committing (new_x, new_y) implies implausible speed."""
        speed = self._windowed_speed_m_s(new_x, new_y, now)
        if speed is None:
            return False
        return speed > self._cfg.max_implied_speed_m_s

    def _windowed_speed_m_s(self, new_x: float, new_y: float, now: float) -> float | None:
        """Estimate speed over a fixed history window, not one frame."""
        window_s = self._cfg.implied_speed_window_s
        if not math.isfinite(self._cfg.max_implied_speed_m_s) or window_s <= 0.0:
            return None
        reference = self._window_reference(now)
        if reference is None:
            return None
        ref_t, ref_x, ref_y = reference
        dt = now - ref_t
        if dt <= 1e-6:
            return None
        return math.hypot(new_x - ref_x, new_y - ref_y) / dt

    def _window_reference(self, now: float) -> tuple[float, float, float] | None:
        """Most recent accepted pose at least `implied_speed_window_s` old."""
        if not self._commit_history:
            return None
        max_age = max(self._cfg.implied_speed_window_s * 2.0, self._cfg.implied_speed_window_s + 1e-3)
        while len(self._commit_history) >= 2 and self._commit_history[1][0] < now - max_age:
            self._commit_history.popleft()
        cutoff = now - self._cfg.implied_speed_window_s
        for sample in reversed(self._commit_history):
            if sample[0] <= cutoff:
                return sample
        return None

    def _record_commit(self, now: float) -> None:
        self._commit_history.append((now, self._x, self._y))
        max_age = max(self._cfg.implied_speed_window_s * 2.0, self._cfg.implied_speed_window_s + 1e-3)
        while len(self._commit_history) >= 2 and self._commit_history[1][0] < now - max_age:
            self._commit_history.popleft()

    def _match_common_bin_translation(
        self,
        scan: Scan2D,
        gyro_delta: float,
    ) -> MatchResult:
        """Estimate forward translation from common scan bins under gyro yaw."""
        if self._keyframe_scan is None:
            return MatchResult(
                dx=0.0,
                dy=0.0,
                dyaw=gyro_delta,
                mean_residual=float("inf"),
                converged=False,
                n_correspondences=0,
                iterations=0,
            )

        valid = (
            (self._keyframe_scan.ranges < self._keyframe_scan.config.max_range_m)
            & (scan.ranges < scan.config.max_range_m)
            & (np.abs(scan.angles) <= self._cfg.common_bin_max_angle_rad)
        )
        n_valid = int(valid.sum())
        if n_valid < self._cfg.min_correspondences:
            return MatchResult(
                dx=0.0,
                dy=0.0,
                dyaw=gyro_delta,
                mean_residual=float("inf"),
                converged=False,
                n_correspondences=n_valid,
                iterations=1,
            )

        ref_x = self._keyframe_scan.ranges[valid] * np.cos(self._keyframe_scan.angles[valid])
        src_x = scan.ranges[valid] * np.cos(scan.angles[valid])
        keep_n = int(self._cfg.common_bin_keep_nearest_n)
        if 0 < keep_n < n_valid:
            # Prefer the nearest front-facing structure. Farther bins in
            # the same angular band are often side walls or open space,
            # and on monocular depth those were a major source of
            # wrong-sign forward estimates on long straight drives.
            nearest_order = np.argsort(np.minimum(ref_x, src_x))
            keep = nearest_order[:keep_n]
            ref_x = ref_x[keep]
            src_x = src_x[keep]
        diffs = ref_x - src_x
        dx = float(np.median(diffs))
        residual = float(np.median(np.abs(diffs - dx)))
        return MatchResult(
            dx=dx,
            dy=0.0,
            dyaw=gyro_delta,
            mean_residual=residual,
            converged=True,
            n_correspondences=n_valid,
            iterations=1,
        )

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
        self._last_gyro_yaw = None
        self._last_update_t = None
        self._consecutive_implausible = 0
        self._consecutive_rotating = 0
        self._commit_history.clear()
        self._motion_allowed_until = float("-inf")
        self._icp_dx_window.clear()
        self._icp_dy_window.clear()
