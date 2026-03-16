"""Infrared line-following sensors (3x digital).

PCB v1: GPIO 16, 20, 21
PCB v2: GPIO 16, 26, 21

Returns a 3-bit value (0-7) representing the combined sensor state.
"""

from __future__ import annotations

import logging
from gpiozero import LineSensor

log = logging.getLogger(__name__)

_PINS_V1 = (16, 20, 21)
_PINS_V2 = (16, 26, 21)


class InfraredSensors:
    def __init__(self, pcb_version: int = 2) -> None:
        pins = _PINS_V2 if pcb_version == 2 else _PINS_V1
        self._sensors = [LineSensor(pin) for pin in pins]
        log.info("IR sensors initialised on pins %s", pins)

    def read(self) -> int:
        """Return 3-bit combined value: (IR1 << 2) | (IR2 << 1) | IR3."""
        bits = [1 if s.value else 0 for s in self._sensors]
        return (bits[0] << 2) | (bits[1] << 1) | bits[2]

    def close(self) -> None:
        for s in self._sensors:
            s.close()
