"""
Protocol definitions for the Freenove tank robot.

Legacy protocol (Android app compatible):
  - Command port 5003: text-based "CMD_NAME#param1#param2#...\n"
  - Video port 8003: [4-byte LE uint32 length][JPEG frame data] repeating

Modern protocol (WebSocket):
  - JSON messages over WebSocket for commands + telemetry
  - Binary WebSocket frames for video
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Legacy protocol constants (must match Freenove Android app)
# ---------------------------------------------------------------------------
LEGACY_CMD_PORT = 5003
LEGACY_VIDEO_PORT = 8003
LEGACY_SEPARATOR = "#"
LEGACY_FRAME_HEADER = struct.Struct("<I")  # 4-byte little-endian uint32


class Cmd(str, Enum):
    """Command identifiers — values must match the Android app exactly."""
    MOTOR = "CMD_MOTOR"
    SERVO = "CMD_SERVO"
    LED = "CMD_LED"
    SONIC = "CMD_SONIC"
    MODE = "CMD_MODE"
    ACTION = "CMD_ACTION"


class CarMode(int, Enum):
    MANUAL = 1
    SONAR = 2
    INFRARED = 3
    CLAMP_STOP = 4
    CLAMP_UP = 5
    CLAMP_DOWN = 6


class LedEffect(int, Enum):
    OFF = 0
    STATIC = 1
    COLOR_WIPE = 2
    BLINK = 3
    BREATHE = 4
    RAINBOW = 5


# ---------------------------------------------------------------------------
# Parsed message
# ---------------------------------------------------------------------------
@dataclass
class Message:
    command: str
    int_params: list[int] = field(default_factory=list)
    raw: str = ""

    @staticmethod
    def parse(raw: str) -> Message:
        """Parse a legacy protocol message like 'CMD_MOTOR#2000#-2000'."""
        raw = raw.strip()
        parts = [p for p in raw.split(LEGACY_SEPARATOR) if p]
        command = parts[0] if parts else ""
        int_params: list[int] = []
        for p in parts[1:]:
            try:
                int_params.append(round(float(p)))
            except ValueError:
                pass
        return Message(command=command, int_params=int_params, raw=raw)

    def encode(self) -> str:
        """Encode back to legacy wire format."""
        if self.int_params:
            params = LEGACY_SEPARATOR.join(str(p) for p in self.int_params)
            return f"{self.command}{LEGACY_SEPARATOR}{params}"
        return self.command


# ---------------------------------------------------------------------------
# Motor limits
# ---------------------------------------------------------------------------
MOTOR_DUTY_MIN = -4095
MOTOR_DUTY_MAX = 4095

# Servo angle limits per channel
SERVO_LIMITS: dict[int, tuple[int, int]] = {
    0: (90, 140),
    1: (75, 150),
    2: (0, 180),
}

# Default servo positions
SERVO_DEFAULTS: dict[int, int] = {
    0: 130,   # grabber slightly open
    1: 140,   # arm up
}


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def clamp_duty(duty: int) -> int:
    return clamp(duty, MOTOR_DUTY_MIN, MOTOR_DUTY_MAX)


def clamp_servo(channel: int, angle: int) -> int:
    lo, hi = SERVO_LIMITS.get(channel, (0, 180))
    return clamp(angle, lo, hi)
