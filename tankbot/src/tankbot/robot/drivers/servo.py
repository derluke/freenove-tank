"""Servo controller supporting PCB v1 (gpiozero) and PCB v2 (hardware PWM).

Channels:
  0 → GPIO 7   — grabber (90° open, 150° closed, default 90°)
  1 → GPIO 8   — arm     (90° down, 150° up, default 140°)
  2 → GPIO 25  — camera  (0-180°)

Includes smooth sweeping — servos move 1° per tick instead of jumping.
"""

from __future__ import annotations

import asyncio
import logging
from tankbot.shared.protocol import clamp_servo, SERVO_DEFAULTS

log = logging.getLogger(__name__)

# GPIO pins for each servo channel
_PINS = {0: 7, 1: 8, 2: 25}

# Sweep config
_SWEEP_INTERVAL = 0.02  # 50 Hz — one step every 20ms
_SWEEP_STEP = 3         # degrees per tick (~150°/sec)
_DETACH_DELAY = 0.5     # seconds after reaching target before cutting PWM


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

    def detach(self, channel: int) -> None:
        """Stop sending PWM so servo draws no current."""
        if channel in self._servos:
            self._servos[channel].angle = None  # type: ignore[assignment]

    def stop(self) -> None:
        pass

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
        from gpiozero import AngularServo

        self._servo2 = AngularServo(
            25, initial_angle=0, min_angle=0, max_angle=180,
            min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000,
        )

    def set_angle(self, channel: int, angle: int) -> None:
        if channel in self._pwms:
            duty = 2.5 + (angle / 180) * 10.0
            self._pwms[channel].change_duty_cycle(duty)
        elif channel == 2:
            self._servo2.angle = angle

    def detach(self, channel: int) -> None:
        """Stop sending PWM so servo draws no current."""
        if channel in self._pwms:
            self._pwms[channel].change_duty_cycle(0)
        elif channel == 2:
            self._servo2.angle = None  # type: ignore[assignment]

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

        self._angles: dict[int, int] = {}
        self._targets: dict[int, int] = {}
        self._detached: set[int] = set()
        self._sweep_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._detach_task: asyncio.Task | None = None  # type: ignore[type-arg]

        # Move to default positions (immediate — no sweep on startup)
        for ch, angle in SERVO_DEFAULTS.items():
            angle = clamp_servo(ch, angle)
            self._angles[ch] = angle
            self._targets[ch] = angle
            self._backend.set_angle(ch, angle)

        log.info("Servos initialised (pcb=%d, pi=%d)", pcb_version, pi_version)

    def set_angle(self, channel: int, angle: int) -> None:
        """Set target angle — servo sweeps smoothly to it."""
        angle = clamp_servo(channel, angle)
        self._targets[channel] = angle
        self._ensure_sweep_running()

    def get_angle(self, channel: int) -> int:
        """Get current actual angle (may still be sweeping toward target)."""
        return self._angles.get(channel, SERVO_DEFAULTS.get(channel, 90))

    def get_target(self, channel: int) -> int:
        """Get target angle."""
        return self._targets.get(channel, SERVO_DEFAULTS.get(channel, 90))

    def stop(self) -> None:
        self._backend.stop()

    def close(self) -> None:
        # Don't move or release PWM — let servos hold position.
        # os._exit() cleans up the pins.
        pass

    def _ensure_sweep_running(self) -> None:
        # Cancel any pending detach — servo is active again
        if self._detach_task is not None and not self._detach_task.done():
            self._detach_task.cancel()
        if self._sweep_task is None or self._sweep_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._sweep_task = loop.create_task(self._sweep_loop())
            except RuntimeError:
                # No event loop — apply immediately
                for ch, target in self._targets.items():
                    self._angles[ch] = target
                    self._backend.set_angle(ch, target)

    async def _sweep_loop(self) -> None:
        """Smoothly step each servo toward its target."""
        try:
            while True:
                all_done = True
                for ch in list(self._targets):
                    current = self._angles.get(ch, 90)
                    target = self._targets[ch]
                    if current != target:
                        all_done = False
                        self._detached.discard(ch)
                        if current < target:
                            current = min(current + _SWEEP_STEP, target)
                        else:
                            current = max(current - _SWEEP_STEP, target)
                        self._angles[ch] = current
                        self._backend.set_angle(ch, current)
                if all_done:
                    break
                await asyncio.sleep(_SWEEP_INTERVAL)
            # Schedule deferred detach to save power
            loop = asyncio.get_running_loop()
            self._detach_task = loop.create_task(self._deferred_detach())
        except asyncio.CancelledError:
            pass

    async def _deferred_detach(self) -> None:
        """Wait, then cut PWM on idle servos to save power."""
        try:
            await asyncio.sleep(_DETACH_DELAY)
            for ch in list(self._targets):
                if self._angles.get(ch) == self._targets.get(ch) and ch not in self._detached:
                    self._backend.detach(ch)
                    self._detached.add(ch)
                    log.debug("Detached servo %d to save power", ch)
        except asyncio.CancelledError:
            pass
