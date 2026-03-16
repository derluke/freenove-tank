"""DC motor driver for the tank treads.

Left motor:  GPIO 24 (forward), GPIO 23 (backward)
Right motor: GPIO 5  (forward), GPIO 6  (backward)

Duty range: -4095..4095  (mapped to 0..1 for gpiozero)
"""

from __future__ import annotations

import logging
from gpiozero import Motor as GpioMotor

from tankbot.shared.protocol import clamp_duty

log = logging.getLogger(__name__)

_MAX_DUTY = 4096  # divisor to normalise to 0..1


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
        log.info("Motors initialised (L=%d/%d R=%d/%d)", left_forward, left_backward, right_forward, right_backward)

    def set(self, left: int, right: int) -> None:
        left = clamp_duty(left)
        right = clamp_duty(right)
        self._drive(self._left, left)
        self._drive(self._right, right)

    def stop(self) -> None:
        self._left.stop()
        self._right.stop()

    def close(self) -> None:
        self.stop()
        self._left.close()
        self._right.close()

    @staticmethod
    def _drive(motor: GpioMotor, duty: int) -> None:
        if duty > 0:
            motor.forward(duty / _MAX_DUTY)
        elif duty < 0:
            motor.backward(-duty / _MAX_DUTY)
        else:
            motor.stop()
