"""Pose abstraction shared across the autonomy redesign (doc §3.2).

`PoseEstimate` is the contract every pose-consuming layer uses — reactive
safety grid, rolling frontier grid, global consistency. Locking it down
*before* multiple implementations land means the abstraction does not
churn as each new estimator appears.

The concrete `PoseSource` implementations (gyro, scan-match, encoders,
VIO, AprilTag) land incrementally. Phase 1 ships `GyroPoseSource`
(heading only, `xy_m=None`) and keeps `NullPoseSource` as a replay/unit
test stub for no-IMU cases.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator


class HealthState(Enum):
    """Per-tick pose-source health.

    Consumers gate behavior on this, not on a boolean "tracking lost".
    See doc §3.4 for transition criteria; they are implementation-defined
    and set from replay tuning in Phase 0c, not from first principles.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BROKEN = "broken"


@dataclass(frozen=True)
class PoseEstimate:
    """2D pose estimate, absolute since init. Matches doc §3.2 verbatim."""

    # Capture timestamp of the sensor readings this estimate was derived
    # from. Monotonic clock, seconds. Consumers use this to align with
    # depth frames.
    t_monotonic: float

    # Planar pose in a 2D world frame anchored at the robot's init pose.
    # Yaw is continuous (unwrapped, not wrapped to [-pi, pi]), so planner
    # code does not have to special-case wraparound. Always valid from
    # init onward.
    yaw_rad: float
    yaw_var: float  # rad^2, 1-sigma^2 uncertainty

    # Absolute xy since init, meters. None when no translation estimator
    # is online or the current estimator has fallen back to unknown.
    # Incremental deltas are derived by differencing, not primitive.
    xy_m: tuple[float, float] | None
    xy_var: float | None  # m^2, isotropic 1-sigma^2 for v1

    # Health state drives consumer behavior (§3.4).
    health: HealthState

    # Provenance — which estimator produced this. Useful for logging,
    # replay analysis, and consumer heuristics (e.g., planner can prefer
    # VIO output over scan-match output when both are available).
    source: str


class PoseSource(ABC):
    """Factory for `PoseEstimate` instances.

    Implementations own their internal state (integrator, filter, etc.)
    and expose a `latest()` method that returns the most recent estimate.
    They may block briefly to fetch a fresh reading, but must not block
    the control loop for longer than a tick.
    """

    source_name: str = "base"

    @abstractmethod
    def latest(self) -> PoseEstimate:
        """Return the most recent pose estimate. Must be cheap to call."""


class NullPoseSource(PoseSource):
    """Trivial pose source used before an IMU is wired.

    Always reports HEALTHY at yaw=0 with `xy_m=None`. This is valid for the
    reactive safety layer at its target control rate (≥15 Hz) because the
    layer rebuilds the occupancy grid from scratch every tick — nothing is
    persisted across frames, so there is no rotation compensation to do.

    As soon as a real gyro arrives, swap this for `GyroPoseSource` without
    touching any consumer.
    """

    source_name = "null"

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock

    def latest(self) -> PoseEstimate:
        return PoseEstimate(
            t_monotonic=self._clock(),
            yaw_rad=0.0,
            yaw_var=0.0,
            xy_m=None,
            xy_var=None,
            health=HealthState.HEALTHY,
            source=self.source_name,
        )


class GyroPoseSource(PoseSource):
    """Phase 1 pose source: heading from the IMU gyro, no translation.

    `xy_m` is always `None` by design. Health reflects whether gyro
    samples have arrived recently enough to trust rotation compensation
    in the reactive grid.
    """

    source_name = "gyro"

    def __init__(
        self,
        gyro: GyroIntegrator,
        *,
        clock: Callable[[], float] = time.monotonic,
        yaw_var: float = 0.001,
        stale_after_s: float = 0.25,
    ) -> None:
        self._gyro = gyro
        self._clock = clock
        self._yaw_var = yaw_var
        self._stale_after_s = stale_after_s

    def latest(self) -> PoseEstimate:
        now = self._clock()
        last_sample_t = self._gyro.last_sample_t
        healthy = (
            last_sample_t is not None
            and self._gyro.n_samples > 0
            and (now - last_sample_t) <= self._stale_after_s
        )
        return PoseEstimate(
            t_monotonic=now,
            yaw_rad=self._gyro.yaw_rad,
            yaw_var=self._yaw_var if healthy else self._yaw_var * 100.0,
            xy_m=None,
            xy_var=None,
            health=HealthState.HEALTHY if healthy else HealthState.BROKEN,
            source=self.source_name,
        )
