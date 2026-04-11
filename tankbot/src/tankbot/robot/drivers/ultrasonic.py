"""HC-SR04 ultrasonic distance sensor.

Trigger: GPIO 27,  Echo: GPIO 22
Returns distance in cm, or -1 on failure.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class _GpiozeroBackend:
    def __init__(self, trigger: int, echo: int) -> None:
        import warnings

        from gpiozero import DistanceSensor, PWMSoftwareFallback

        warnings.filterwarnings("ignore", category=PWMSoftwareFallback)
        self._sensor = DistanceSensor(echo=echo, trigger=trigger, max_distance=3)

    def get_distance(self) -> float:
        try:
            return round(self._sensor.distance * 100, 1)
        except Exception:
            return -1.0

    def close(self) -> None:
        self._sensor.close()


class _LgpioBackend:
    def __init__(self, trigger: int, echo: int) -> None:
        import lgpio

        self._lgpio = lgpio
        self._trigger = trigger
        self._echo = echo
        try:
            self._chip = lgpio.gpiochip_open(0)
        except Exception:
            self._chip = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(self._chip, self._trigger)
        lgpio.gpio_claim_input(self._chip, self._echo)

    def get_distance(self) -> float:
        lg = self._lgpio
        try:
            lg.gpio_write(self._chip, self._trigger, 0)
            time.sleep(0.05)
            lg.gpio_write(self._chip, self._trigger, 1)
            time.sleep(0.00001)
            lg.gpio_write(self._chip, self._trigger, 0)

            timeout = time.time() + 1.0
            start = time.time()
            while lg.gpio_read(self._chip, self._echo) == 0:
                start = time.time()
                if start > timeout:
                    return -1.0

            stop = time.time()
            while lg.gpio_read(self._chip, self._echo) == 1:
                stop = time.time()
                if stop > timeout:
                    return -1.0

            return round((stop - start) * 34300 / 2, 1)
        except Exception:
            return -1.0

    def close(self) -> None:
        if self._chip is not None:
            try:
                self._lgpio.gpiochip_close(self._chip)
            except Exception:
                pass
            self._chip = None


class Ultrasonic:
    def __init__(self, trigger: int = 27, echo: int = 22, pi_version: int = 2) -> None:
        if pi_version == 2:
            self._backend = _LgpioBackend(trigger, echo)
        else:
            self._backend = _GpiozeroBackend(trigger, echo)
        log.info("Ultrasonic sensor initialised (pi_version=%d)", pi_version)

    def get_distance(self) -> float:
        return self._backend.get_distance()

    def close(self) -> None:
        self._backend.close()
