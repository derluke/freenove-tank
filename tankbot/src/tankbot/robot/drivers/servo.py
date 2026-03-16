"""Servo controller supporting PCB v1 (gpiozero) and PCB v2 (hardware PWM).

Channels:
  0 → GPIO 7   (range 90-150°)
  1 → GPIO 8   (range 90-150°)
  2 → GPIO 25  (range 0-180°)
"""

from __future__ import annotations

import logging
from tankbot.shared.protocol import clamp_servo, SERVO_DEFAULTS

log = logging.getLogger(__name__)

# GPIO pins for each servo channel
_PINS = {0: 7, 1: 8, 2: 25}


class _GpiozeroBackend:
    def __init__(self) -> None:
        from gpiozero import AngularServo

        min_pw = 0.5 / 1000
        max_pw = 2.5 / 1000
        self._servos = {
            ch: AngularServo(
                pin,
                initial_angle=0,
                min_angle=0,
                max_angle=180,
                min_pulse_width=min_pw,
                max_pulse_width=max_pw,
            )
            for ch, pin in _PINS.items()
        }

    def set_angle(self, channel: int, angle: int) -> None:
        if channel in self._servos:
            self._servos[channel].angle = angle

    def stop(self) -> None:
        pass  # gpiozero servos don't need explicit stop

    def close(self) -> None:
        for s in self._servos.values():
            s.close()


class _HardwarePWMBackend:
    """For PCB v2 using rpi_hardware_pwm on channels 0/1 (GPIO 12/13)."""

    def __init__(self, chip: int = 0) -> None:
        from rpi_hardware_pwm import HardwarePWM

        self._pwms = {
            0: HardwarePWM(pwm_channel=0, hz=50, chip=chip),
            1: HardwarePWM(pwm_channel=1, hz=50, chip=chip),
        }
        for pwm in self._pwms.values():
            pwm.start(0)
        # Channel 2 still uses gpiozero on GPIO 25
        from gpiozero import AngularServo

        self._servo2 = AngularServo(
            25, initial_angle=0, min_angle=0, max_angle=180,
            min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000,
        )

    def set_angle(self, channel: int, angle: int) -> None:
        if channel in self._pwms:
            duty = 2.5 + (angle / 180) * 10.0  # 2.5% - 12.5%
            self._pwms[channel].change_duty_cycle(duty)
        elif channel == 2:
            self._servo2.angle = angle

    def stop(self) -> None:
        for pwm in self._pwms.values():
            pwm.stop()

    def close(self) -> None:
        self.stop()
        self._servo2.close()


class ServoController:
    def __init__(self, pcb_version: int = 2, pi_version: int = 2) -> None:
        if pcb_version == 2:
            self._backend = _HardwarePWMBackend()
        else:
            self._backend = _GpiozeroBackend()

        # Move to default positions
        for ch, angle in SERVO_DEFAULTS.items():
            self.set_angle(ch, angle)

        log.info("Servos initialised (pcb=%d, pi=%d)", pcb_version, pi_version)

    def set_angle(self, channel: int, angle: int) -> None:
        angle = clamp_servo(channel, angle)
        self._backend.set_angle(channel, angle)

    def stop(self) -> None:
        self._backend.stop()

    def close(self) -> None:
        self._backend.close()
