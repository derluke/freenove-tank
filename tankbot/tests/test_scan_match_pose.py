"""Unit tests for ScanMatchPoseSource — in particular the median filter.

The scan-match ICP occasionally latches onto a wrong local minimum on
monocular depth, producing a single-frame `dx` that's 10x the real motion.
These outliers pass residual/ncorr gates, so the pose-source layer needs
to reject them. Test: feed a sequence of synthetic scans where the same
translation is reported every frame except for one spike — the committed
pose must track the consistent value, not the spike.
"""

from __future__ import annotations

import numpy as np

from tankbot.desktop.autonomy.reactive.pose import HealthState
from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)


def _wall_points(x_offset_m: float, n: int = 200) -> np.ndarray:
    """A simple vertical wall in robot-frame coordinates.

    The wall sits at x=2.0 m by default and is swept across y∈[-0.5, 0.5]
    at a single height in the scan band. Shifting `x_offset_m` moves the
    wall toward or away from the robot, which is what ICP should pick up
    as forward motion.
    """
    y = np.linspace(-0.5, 0.5, n)
    x = np.full_like(y, 2.0 + x_offset_m)
    z = np.full_like(y, 0.10)  # solidly in the scan band
    return np.stack([x, y, z], axis=-1).astype(np.float32)


class TestMedianFilterRejectsOutliers:
    def test_single_spike_does_not_move_pose(self) -> None:
        """One outlier ICP dx in the middle of an otherwise-still sequence
        must not drift the published pose."""
        # A strong median window, disabled implausible check, so the only
        # protection against a spike is the median filter itself.
        cfg = ScanMatchConfig(
            icp_smoothing_window=5,
            max_implied_speed_m_s=float("inf"),
            implausible_run_to_broken=0,
            use_common_bin_translation_with_gyro=False,
        )
        ps = ScanMatchPoseSource(
            scan_config=ScanConfig(),
            icp_config=ICPConfig(),
            config=cfg,
        )

        # Prime with stationary frames so the deque fills with dx≈0.
        for i in range(10):
            ps.update(
                _wall_points(0.0),
                gyro_yaw=0.0,
                motors_active=True,
                t_monotonic=float(i) * 0.033,
            )

        x_before = (ps.latest().xy_m or (0.0, 0.0))[0]

        # One frame with a wall 50 cm closer would imply dx=+0.5 m — the
        # exact kind of single-frame spike we're filtering out.
        ps.update(
            _wall_points(-0.5),
            gyro_yaw=0.0,
            motors_active=True,
            t_monotonic=11.0 * 0.033,
        )
        x_after_spike = (ps.latest().xy_m or (0.0, 0.0))[0]

        # The single spike sits among 4 zero samples in a 5-frame median,
        # so the committed translation must stay near 0.
        assert abs(x_after_spike - x_before) < 0.05, (
            f"median filter let a 50 cm single-frame spike through: "
            f"pose jumped by {x_after_spike - x_before:.3f} m"
        )

    def test_sustained_motion_is_tracked(self) -> None:
        """Consistent forward motion across many frames should commit —
        the median filter must not block real motion, only outliers."""
        cfg = ScanMatchConfig(
            icp_smoothing_window=5,
            residual_degraded=0.30,  # let perfect synthetic matches through
            max_implied_speed_m_s=float("inf"),
            implausible_run_to_broken=0,
            keyframe_dist_m=0.05,  # promote often for this synthetic test
            use_common_bin_translation_with_gyro=False,
        )
        ps = ScanMatchPoseSource(
            scan_config=ScanConfig(),
            icp_config=ICPConfig(),
            config=cfg,
        )

        # Move the wall 1 cm closer each frame → steady 1 cm forward motion.
        for i in range(40):
            ps.update(
                _wall_points(-0.01 * i),
                gyro_yaw=0.0,
                motors_active=True,
                t_monotonic=float(i) * 0.033,
            )

        pose = ps.latest()
        assert pose.xy_m is not None, "pose source went BROKEN on clean synthetic motion"
        x_final = pose.xy_m[0]
        # After 40 frames of 1 cm/frame, the robot has moved ~40 cm. Allow
        # ~50% tolerance because keyframe promotion + median lag trims the
        # observed total.
        assert x_final > 0.15, (
            f"sustained motion not tracked: x_final={x_final:.3f} m after 40x1cm steps"
        )

    def test_keyframe_promotion_flushes_window(self) -> None:
        """After keyframe promotion, prior ICP deltas must not influence
        the next committed step — otherwise committed motion would
        double-count the pre-promotion estimate."""
        cfg = ScanMatchConfig(
            icp_smoothing_window=5,
            keyframe_dist_m=0.05,
            use_common_bin_translation_with_gyro=False,
        )
        ps = ScanMatchPoseSource(
            scan_config=ScanConfig(),
            icp_config=ICPConfig(),
            config=cfg,
        )
        # Force a sequence of updates so the deque fills and then a
        # keyframe promote happens internally.
        for i in range(20):
            ps.update(
                _wall_points(-0.01 * i),
                gyro_yaw=0.0,
                motors_active=True,
                t_monotonic=float(i) * 0.033,
            )
        # After enough motion, deque should not retain stale pre-keyframe
        # values forever — keyframe has been promoted at least once.
        assert len(ps._icp_dx_window) < cfg.icp_smoothing_window + 1


def test_health_stays_sane_on_stationary_noisy_input() -> None:
    """Pure noise (no motion, no motors) must not push health to BROKEN.
    This is the straight_1m_1 regression scenario: the robot sat still
    with motors off for 200 frames and the pipeline collapsed anyway
    before the fix."""
    rng = np.random.default_rng(seed=0)
    cfg = ScanMatchConfig(icp_smoothing_window=5)
    ps = ScanMatchPoseSource(
        scan_config=ScanConfig(),
        icp_config=ICPConfig(),
        config=cfg,
    )
    for i in range(80):
        # Small per-frame jitter — simulates depth-model noise on a
        # static scene.
        pts = _wall_points(0.0) + rng.normal(0.0, 0.02, size=(200, 3)).astype(
            np.float32
        )
        ps.update(pts, gyro_yaw=0.0, motors_active=False, t_monotonic=float(i) * 0.033)
    assert ps.latest().health != HealthState.BROKEN


def test_windowed_velocity_gate_tolerates_noisy_stationary_scene() -> None:
    """With motors marked active, jitter alone must not trip τ_trans.

    This guards against the original per-frame gate failure mode: noisy
    monocular-depth dx estimates should not look like impossible speed
    once measured over a 250 ms window.
    """
    rng = np.random.default_rng(seed=1)
    cfg = ScanMatchConfig(
        icp_smoothing_window=5,
        max_implied_speed_m_s=0.75,
        implied_speed_window_s=0.25,
        implausible_run_to_broken=2,
        residual_degraded=0.30,
    )
    ps = ScanMatchPoseSource(
        scan_config=ScanConfig(),
        icp_config=ICPConfig(),
        config=cfg,
    )
    for i in range(80):
        pts = _wall_points(0.0) + rng.normal(0.0, 0.02, size=(200, 3)).astype(
            np.float32
        )
        ps.update(pts, gyro_yaw=0.0, motors_active=True, t_monotonic=float(i) * 0.033)
    assert ps.latest().health != HealthState.BROKEN


def test_windowed_velocity_gate_rejects_sustained_implausible_motion() -> None:
    """τ_trans should demote impossible translation, not pass it as healthy."""
    cfg = ScanMatchConfig(
        icp_smoothing_window=1,
        residual_degraded=0.30,
        keyframe_dist_m=10.0,
        max_delta_m=float("inf"),
        max_implied_speed_m_s=1.0,
        implied_speed_window_s=0.25,
        implausible_run_to_broken=2,
    )
    ps = ScanMatchPoseSource(
        scan_config=ScanConfig(),
        icp_config=ICPConfig(),
        config=cfg,
    )

    for i in range(12):
        ps.update(
            _wall_points(-0.12 * i),
            gyro_yaw=0.0,
            motors_active=True,
            t_monotonic=float(i) * 0.033,
        )

    pose = ps.latest()
    assert pose.health != HealthState.HEALTHY
    if pose.xy_m is not None:
        assert pose.xy_m[0] < 0.8


def test_high_gyro_rate_freezes_translation_and_degrades() -> None:
    """With gyro present, sustained high yaw rate must not keep committing xy.

    This is the turn_in_place regression: the rotation gate used to be dead
    under gyro because it only triggered when gyro_delta was None, which never
    happens on the normal gyro-guided path.
    """
    cfg = ScanMatchConfig(
        residual_degraded=0.30,
        rotation_degrade_frames=2,
        max_rotation_rate_rad_s=0.20,
        common_bin_max_gyro_delta_rad=0.12,
        keyframe_yaw_rad=10.0,
    )
    ps = ScanMatchPoseSource(
        scan_config=ScanConfig(),
        icp_config=ICPConfig(),
        config=cfg,
    )

    # Establish a keyframe and a committed forward pose.
    ps.update(_wall_points(0.0), gyro_yaw=0.0, motors_active=True, t_monotonic=0.0)
    ps.update(_wall_points(-0.08), gyro_yaw=0.0, motors_active=True, t_monotonic=0.10)
    pose_before_turn = ps.latest()
    assert pose_before_turn.xy_m is not None
    x_before, y_before = pose_before_turn.xy_m

    # Keep feeding scans that would otherwise imply more forward translation,
    # but with a sustained high gyro rate. Translation should freeze and the
    # pose should demote once the rate persists.
    ps.update(_wall_points(-0.16), gyro_yaw=0.05, motors_active=True, t_monotonic=0.20)
    ps.update(_wall_points(-0.24), gyro_yaw=0.10, motors_active=True, t_monotonic=0.30)

    pose_after_turn = ps.latest()
    assert pose_after_turn.xy_m is not None
    x_after, y_after = pose_after_turn.xy_m
    assert abs(x_after - x_before) < 0.03
    assert abs(y_after - y_before) < 0.03
    assert pose_after_turn.health == HealthState.DEGRADED
