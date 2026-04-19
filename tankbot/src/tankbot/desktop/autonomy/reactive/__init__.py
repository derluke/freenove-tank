"""Reactive safety layer (Phase 1 of the autonomy redesign).

Entry points:
- `PoseEstimate`, `PoseSource`, `NullPoseSource`, `GyroPoseSource` —
  pose abstraction shared with later phases (§3.2 of
  docs/autonomy_perception_redesign.md).
- `CameraExtrinsics` — one-time static calibration consumed by projection.
- `project_depth_to_robot_points` — depth + extrinsics + intrinsics → robot
  frame point cloud.
- `ReactiveGrid` — robot-centric 2D occupancy grid with stop-zone queries.
- `ReactiveSafety` — orchestrator that turns depth + ultrasonic + pose into
  a motor command with veto rationale.
"""

from __future__ import annotations

from tankbot.desktop.autonomy.reactive.extrinsics import (
    CameraExtrinsics,
    load_extrinsics,
)
from tankbot.desktop.autonomy.reactive.grid import ReactiveGrid, StopZoneQuery
from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator, unwrap_delta
from tankbot.desktop.autonomy.reactive.intrinsics import (
    CameraIntrinsics,
    load_intrinsics,
)
from tankbot.desktop.autonomy.reactive.policy import (
    PolicyDecision,
    ReactiveWanderPolicy,
    WanderState,
)
from tankbot.desktop.autonomy.reactive.pose import (
    GyroPoseSource,
    HealthState,
    NullPoseSource,
    PoseEstimate,
    PoseSource,
)
from tankbot.desktop.autonomy.reactive.projection import project_depth_to_robot_points
from tankbot.desktop.autonomy.reactive.safety import (
    ReactiveSafety,
    SafetyTick,
)
from tankbot.desktop.autonomy.reactive.scan import Scan2D, ScanConfig, extract_scan
from tankbot.desktop.autonomy.reactive.scan_match import (
    ICPConfig,
    MatchResult,
    match_scans,
)
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)

__all__ = [
    "CameraExtrinsics",
    "CameraIntrinsics",
    "GyroIntegrator",
    "GyroPoseSource",
    "HealthState",
    "ICPConfig",
    "MatchResult",
    "NullPoseSource",
    "PolicyDecision",
    "PoseEstimate",
    "PoseSource",
    "ReactiveGrid",
    "ReactiveSafety",
    "ReactiveWanderPolicy",
    "SafetyTick",
    "Scan2D",
    "ScanConfig",
    "ScanMatchConfig",
    "ScanMatchPoseSource",
    "StopZoneQuery",
    "WanderState",
    "extract_scan",
    "load_extrinsics",
    "load_intrinsics",
    "match_scans",
    "project_depth_to_robot_points",
    "unwrap_delta",
]
