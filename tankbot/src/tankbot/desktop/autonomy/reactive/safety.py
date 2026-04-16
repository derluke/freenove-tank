"""Reactive safety orchestrator — depth + ultrasonic + pose → motor command.

`ReactiveSafety` is the top-level entry point for the Phase 1 reactive
layer. Each tick it:

1. Projects the latest depth map into robot-frame 3D points.
2. Builds a fresh robot-centric occupancy grid from those points.
3. Optionally fuses an ultrasonic reading as a short-range confirmation
   or free-space veto.
4. Queries the stop zone and directional clearances.
5. Feeds the queries to `ReactiveWanderPolicy` to get a motor intent.
6. Packages everything into a `SafetyTick` for logging / replay viz.

The orchestrator is *not* responsible for actually sending motor commands
— it returns a `SafetyTick` and the caller decides whether to act on it
(live loop) or just log it (replay harness).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from tankbot.desktop.autonomy.reactive.extrinsics import CameraExtrinsics
from tankbot.desktop.autonomy.reactive.grid import ReactiveGrid, StopZoneQuery
from tankbot.desktop.autonomy.reactive.intrinsics import CameraIntrinsics
from tankbot.desktop.autonomy.reactive.policy import (
    PolicyDecision,
    ReactiveWanderPolicy,
    WanderState,
)
from tankbot.desktop.autonomy.reactive.pose import PoseEstimate, PoseSource
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
    project_depth_to_robot_points,
)


@dataclass(frozen=True)
class SafetyTick:
    """Full decision trace for one reactive tick.

    Designed for replay visualization: every field the annotator needs
    to render a frame + depth + grid + motor intent + veto state is
    here. Also serves as the structured log record for live telemetry.
    """

    t_monotonic: float
    pose: PoseEstimate

    # Grid state.
    n_obstacle_points: int
    stop_zone_blocked: bool
    stop_zone_count: int

    # Directional clearance samples (meters).
    clearance_forward_m: float
    clearance_left_m: float
    clearance_right_m: float

    # Ultrasonic reading, if available.
    ultrasonic_m: float | None

    # Policy output.
    decision: PolicyDecision

    # Timing.
    projection_ms: float
    grid_ms: float
    total_ms: float


# Stale-depth deadline: if no new depth frame arrives within this many
# seconds, the safety layer forces a hard stop. Per Phase 1 behavioral
# limits: "hard forward stop if depth has not updated for >300 ms."
_STALE_DEPTH_DEADLINE_S: float = 0.30

# Ultrasonic confirmation range: if the ultrasonic returns a reading
# below this distance, force stop-zone blocked regardless of the grid.
# Useful as a last-resort backup, especially on hard flat surfaces where
# the ultrasonic is reliable (see feedback_ultrasonic_unreliable.md).
_ULTRASONIC_VETO_M: float = 0.20


class ReactiveSafety:
    """Orchestrates one reactive tick per call to `tick()`."""

    def __init__(
        self,
        *,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        pose_source: PoseSource,
        grid: ReactiveGrid | None = None,
        stop_query: StopZoneQuery | None = None,
        projection_config: ProjectionConfig | None = None,
        policy: ReactiveWanderPolicy | None = None,
    ) -> None:
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._pose_source = pose_source
        self._grid = grid or ReactiveGrid()
        self._stop_query = stop_query or StopZoneQuery()
        self._proj_cfg = projection_config or ProjectionConfig()
        self._policy = policy or ReactiveWanderPolicy()

        # Wander state persists across ticks.
        self._wander_state = WanderState.STOPPED

        # Track the last time we received a valid depth frame.
        self._last_depth_t: float | None = None

    @property
    def wander_state(self) -> WanderState:
        return self._wander_state

    def tick(
        self,
        depth_m: np.ndarray | None,
        *,
        ultrasonic_m: float | None = None,
        t_now: float | None = None,
    ) -> SafetyTick:
        """Run one reactive cycle. Returns a full `SafetyTick`.

        Parameters
        ----------
        depth_m:
            HxW float32 metric depth map, or None if no new frame is
            available this tick (the stale-depth deadline logic handles
            the fallback).
        ultrasonic_m:
            Latest ultrasonic reading in meters, or None if unavailable.
        t_now:
            Monotonic timestamp for this tick.  Defaults to
            ``time.monotonic()``.
        """
        t0 = time.monotonic()
        now = t_now if t_now is not None else t0

        pose = self._pose_source.latest()

        # --- depth staleness check ---
        have_depth = depth_m is not None
        if have_depth:
            self._last_depth_t = now

        depth_stale = self._last_depth_t is None or (now - self._last_depth_t) > _STALE_DEPTH_DEADLINE_S

        # --- projection + grid ---
        t_proj_start = time.monotonic()
        if have_depth:
            points = project_depth_to_robot_points(
                depth_m,
                self._intrinsics,
                self._extrinsics,
                self._proj_cfg,  # type: ignore[arg-type]
            )
            n_points = points.shape[0]
        else:
            points = np.empty((0, 3), dtype=np.float32)
            n_points = 0
        t_proj_end = time.monotonic()

        t_grid_start = time.monotonic()
        self._grid.build_from_points(points)
        t_grid_end = time.monotonic()

        # --- stop-zone query ---
        stop_blocked, stop_count = self._grid.stop_zone_occupied(self._stop_query)

        # Ultrasonic veto: short-range override.
        if ultrasonic_m is not None and ultrasonic_m < _ULTRASONIC_VETO_M:
            stop_blocked = True

        # Stale depth → hard stop.
        if depth_stale:
            stop_blocked = True

        # --- directional clearance ---
        clearance_fwd = self._grid.clearance_in_direction(0.0)
        clearance_left = self._grid.clearance_in_direction(math.pi / 2)
        clearance_right = self._grid.clearance_in_direction(-math.pi / 2)

        # --- policy ---
        if depth_stale and not have_depth:
            decision = ReactiveWanderPolicy.stopped("stale depth — hard stop")
        else:
            decision = self._policy.decide(
                stop_zone_blocked=stop_blocked,
                clearance_left_m=clearance_left,
                clearance_right_m=clearance_right,
                clearance_forward_m=clearance_fwd,
                state=self._wander_state,
            )

        self._wander_state = decision.state

        t_end = time.monotonic()
        return SafetyTick(
            t_monotonic=now,
            pose=pose,
            n_obstacle_points=n_points,
            stop_zone_blocked=stop_blocked,
            stop_zone_count=stop_count,
            clearance_forward_m=clearance_fwd,
            clearance_left_m=clearance_left,
            clearance_right_m=clearance_right,
            ultrasonic_m=ultrasonic_m,
            decision=decision,
            projection_ms=(t_proj_end - t_proj_start) * 1000.0,
            grid_ms=(t_grid_end - t_grid_start) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
        )
