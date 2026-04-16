"""Camera intrinsics loader.

Reads the MASt3R-SLAM-compatible YAML produced by
`tankbot-calibrate-camera`:

    width: 640
    height: 480
    calibration: [fx, fy, cx, cy, k1, k2, p1, p2, k3]

and exposes a typed `CameraIntrinsics` dataclass for the reactive layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, ...]  # (k1, k2, p1, p2, k3)

    @property
    def has_distortion(self) -> bool:
        return any(abs(c) > 1e-9 for c in self.dist_coeffs)


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    calib = list(data["calibration"])
    if len(calib) < 4:
        msg = f"calibration array too short (got {len(calib)}, need at least 4)"
        raise ValueError(msg)
    fx, fy, cx, cy = (float(x) for x in calib[:4])
    dist_coeffs = tuple(float(x) for x in calib[4:])
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=dist_coeffs,
    )
