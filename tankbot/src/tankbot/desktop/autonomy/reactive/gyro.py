"""Gyro-Z yaw integrator.

Consumes calibrated gyro-Z samples (degrees/sec) and accumulates an
absolute yaw angle by trapezoidal integration. The accumulator wraps
via `math` so downstream code can diff two yaws and wrap the result to
(-pi, pi] — see `unwrap_delta()`.

Coordinates: robot body-frame +Z points up, so positive gz = CCW turn
(standard right-hand rule). Scan-match rotations are also CCW-positive,
so gyro yaw feeds directly into ICP as a rotation prior without sign
flips.

Drift: an uncalibrated MPU-6050 can drift tens of degrees per minute;
with the Imu driver's startup calibration + adaptive bias update the
short-term drift over a few keyframes (<1 s) is small, which is all
scan-match needs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class GyroIntegrator:
    """Integrates gyro-Z (dps) into an absolute yaw angle (radians)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._yaw_rad: float = 0.0
        self._last_t: float | None = None
        self._n_samples: int = 0

    def add_sample(self, gz_dps: float, t: float | None = None) -> None:
        """Feed one gyro-Z reading in degrees per second."""
        now = t if t is not None else self._clock()
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        # Guard against clock jumps or very long gaps (>200 ms) — skip
        # integration rather than accumulate garbage.
        if 0.0 < dt < 0.2:
            self._yaw_rad += math.radians(gz_dps) * dt
            self._n_samples += 1
        self._last_t = now

    @property
    def yaw_rad(self) -> float:
        """Current accumulated yaw. May exceed (-pi, pi]."""
        return self._yaw_rad

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def reset(self, yaw_rad: float = 0.0) -> None:
        self._yaw_rad = yaw_rad
        self._last_t = None


def unwrap_delta(delta_rad: float) -> float:
    """Wrap a yaw delta to (-pi, pi]."""
    return math.atan2(math.sin(delta_rad), math.cos(delta_rad))
