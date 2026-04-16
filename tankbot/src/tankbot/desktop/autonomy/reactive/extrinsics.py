"""Camera extrinsics — fixed mounting of the camera on the robot chassis.

The Freenove tank's camera is mechanically fixed to the chassis (the tilt
servo is locked), so the extrinsics are a single one-time measurement.
The reactive layer's floor-plane projection reads these values once at
startup; they are not updated at runtime.

Frame conventions:
- **Robot frame:** +x forward, +y left, +z up. Origin at the chassis
  center at floor level.
- **Camera frame:** +x right, +y down, +z forward (OpenCV convention).

Measured values you should provide:
- `height_m`:  camera optical center above the floor, measured with a ruler.
- `pitch_deg`: positive = camera tilted downward (nose pointed at the
  floor). Measured with a level / phone protractor app.
- `yaw_deg`:   positive = camera rotated left relative to robot forward.
  Usually 0 for a centered mount.
- `roll_deg`:  positive = camera rotated so its +x (right) tilts upward.
  Usually 0.
- `x_offset_m`: how far in front of the chassis center the optical center
  sits.
- `y_offset_m`: how far to the left (+) or right (-) the optical center
  sits relative to chassis center.

The defaults in `calibration/camera_extrinsics.yaml` are **eyeball
estimates**. Replace them with measured values before trusting the
reactive layer in the live loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from the camera frame to the robot frame."""

    height_m: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    x_offset_m: float
    y_offset_m: float

    def rotation_robot_from_camera(self) -> np.ndarray:
        """3x3 rotation that maps camera-frame vectors into robot frame.

        Composition order: base (axis relabel) → pitch → yaw → roll.
        Pitch is around the robot's y axis (left), yaw around z (up),
        roll around x (forward).
        """
        # Base axis relabel: camera (x right, y down, z fwd) → robot (x
        # fwd, y left, z up) with no tilt/yaw/roll.
        r_base = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

        p = math.radians(self.pitch_deg)
        y = math.radians(self.yaw_deg)
        r = math.radians(self.roll_deg)

        r_pitch = np.array(
            [
                [math.cos(p), 0.0, math.sin(p)],
                [0.0, 1.0, 0.0],
                [-math.sin(p), 0.0, math.cos(p)],
            ],
            dtype=np.float64,
        )
        r_yaw = np.array(
            [
                [math.cos(y), -math.sin(y), 0.0],
                [math.sin(y), math.cos(y), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        r_roll = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(r), -math.sin(r)],
                [0.0, math.sin(r), math.cos(r)],
            ],
            dtype=np.float64,
        )
        return r_roll @ r_yaw @ r_pitch @ r_base

    def translation_robot_from_camera(self) -> np.ndarray:
        return np.array([self.x_offset_m, self.y_offset_m, self.height_m], dtype=np.float64)


def load_extrinsics(path: str | Path) -> CameraExtrinsics:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CameraExtrinsics(
        height_m=float(data.get("height_m", 0.12)),
        pitch_deg=float(data.get("pitch_deg", 8.0)),
        yaw_deg=float(data.get("yaw_deg", 0.0)),
        roll_deg=float(data.get("roll_deg", 0.0)),
        x_offset_m=float(data.get("x_offset_m", 0.08)),
        y_offset_m=float(data.get("y_offset_m", 0.0)),
    )
