"""DC motor driver for the tank treads.

Left motor:  GPIO 24 (forward), GPIO 23 (backward)
Right motor: GPIO 5  (forward), GPIO 6  (backward)

Duty range: -4095..4095  (mapped to 0..1 for gpiozero)

Includes software ramping to avoid current spikes that can brownout the Pi.
"""

from __future__ import annotations

import asyncio
import logging
from gpiozero import Motor as GpioMotor

from tankbot.shared.protocol import clamp_duty

log = logging.getLogger(__name__)

_MAX_DUTY = 4096  # divisor to normalise to 0..1

# Ramping config
_RAMP_STEP = 400       # duty units per tick
_RAMP_INTERVAL = 0.02  # seconds between ramp ticks (~50 Hz)


class Motor:
    def __init__(
        self,
        left_forward: int = 24,
        left_backward: int = 23,
        right_forward: int = 5,
        right_backward: int = 6,
    ):
        self._left = GpioMotor(left_forward, left_backward)
        self._right = GpioMotor(right_forward, right_backward)

        # Current actual duty and target duty
        self._current_left = 0
        self._current_right = 0
        self._target_left = 0
        self._target_right = 0

        self._ramp_task: asyncio.Task | None = None  # type: ignore[type-arg]
        log.info("Motors initialised (L=%d/%d R=%d/%d)", left_forward, left_backward, right_forward, right_backward)

    def set(self, left: int, right: int) -> None:
        """Set target speed — ramping will smoothly transition to it."""
        self._target_left = clamp_duty(left)
        self._target_right = clamp_duty(right)
        self._ensure_ramp_running()

    def stop(self) -> None:
        """Set target to zero (ramps down)."""
        self._target_left = 0
        self._target_right = 0
        self._ensure_ramp_running()

    def stop_immediate(self) -> None:
        """Hard stop — no ramping. Use for emergency/shutdown."""
        self._target_left = 0
        self._target_right = 0
        self._current_left = 0
        self._current_right = 0
        self._left.stop()
        self._right.stop()

    def close(self) -> None:
        self.stop_immediate()
        if self._ramp_task and not self._ramp_task.done():
            self._ramp_task.cancel()
        self._left.close()
        self._right.close()

    def _ensure_ramp_running(self) -> None:
        """Start the ramp loop if it's not already running."""
        if self._ramp_task is None or self._ramp_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._ramp_task = loop.create_task(self._ramp_loop())
            except RuntimeError:
                # No event loop — apply immediately (e.g. during shutdown)
                self._apply_immediate()

    async def _ramp_loop(self) -> None:
        """Smoothly ramp current duty toward target duty."""
        try:
            while self._current_left != self._target_left or self._current_right != self._target_right:
                self._current_left = _step_toward(self._current_left, self._target_left, _RAMP_STEP)
                self._current_right = _step_toward(self._current_right, self._target_right, _RAMP_STEP)
                self._drive(self._left, self._current_left)
                self._drive(self._right, self._current_right)
                await asyncio.sleep(_RAMP_INTERVAL)
        except asyncio.CancelledError:
            pass

    def _apply_immediate(self) -> None:
        """Apply target directly (fallback when no event loop)."""
        self._current_left = self._target_left
        self._current_right = self._target_right
        self._drive(self._left, self._current_left)
        self._drive(self._right, self._current_right)

    @staticmethod
    def _drive(motor: GpioMotor, duty: int) -> None:
        if duty > 0:
            motor.forward(duty / _MAX_DUTY)
        elif duty < 0:
            motor.backward(-duty / _MAX_DUTY)
        else:
            motor.stop()


def _step_toward(current: int, target: int, step: int) -> int:
    """Move current toward target by at most step."""
    diff = target - current
    if abs(diff) <= step:
        return target
    return current + (step if diff > 0 else -step)
