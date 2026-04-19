from __future__ import annotations

import math

from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator
from tankbot.desktop.autonomy.reactive.pose import GyroPoseSource, HealthState


def test_gyro_pose_source_reports_yaw_without_translation() -> None:
    now = 0.0

    def clock() -> float:
        return now

    gyro = GyroIntegrator(clock=clock)
    pose_source = GyroPoseSource(gyro, clock=clock, stale_after_s=0.25)

    gyro.add_sample(90.0, t=0.00)
    gyro.add_sample(90.0, t=0.10)
    now = 0.11

    pose = pose_source.latest()
    assert pose.health == HealthState.HEALTHY
    assert pose.xy_m is None
    assert math.isclose(pose.yaw_rad, math.radians(9.0), abs_tol=1e-6)


def test_gyro_pose_source_goes_broken_when_imu_stales() -> None:
    now = 0.0

    def clock() -> float:
        return now

    gyro = GyroIntegrator(clock=clock)
    pose_source = GyroPoseSource(gyro, clock=clock, stale_after_s=0.25)

    gyro.add_sample(0.0, t=0.00)
    gyro.add_sample(0.0, t=0.05)
    now = 0.50

    pose = pose_source.latest()
    assert pose.health == HealthState.BROKEN
    assert pose.xy_m is None
