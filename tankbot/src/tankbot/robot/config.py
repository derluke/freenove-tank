"""Robot configuration — auto-detects Pi and PCB version."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PARAMS_FILE = Path("params.json")


def _detect_pi_version() -> int:
    try:
        model = Path("/sys/firmware/devicetree/base/model").read_text()
        return 2 if "Raspberry Pi 5" in model else 1
    except Exception:
        return 1


@dataclass(frozen=True)
class RobotConfig:
    pcb_version: int = 2
    pi_version: int = 2
    cmd_port: int = 5003
    video_port: int = 8003
    ws_port: int = 9000
    stream_size: tuple[int, int] = (640, 480)

    @staticmethod
    def load(path: Path = _PARAMS_FILE) -> RobotConfig:
        pi_version = _detect_pi_version()
        pcb_version = 2  # default

        if path.exists():
            try:
                data = json.loads(path.read_text())
                pcb_version = data.get("Pcb_Version", 2)
            except (json.JSONDecodeError, OSError):
                pass

        return RobotConfig(pcb_version=pcb_version, pi_version=pi_version)
