"""Vision pipeline for low-level autonomous driving.

Connects to the robot via WebSocket, receives frames, runs MASt3R-SLAM
for dense mapping and depth estimation, and sends motor commands back to
navigate autonomously.

Behaviour: cautious exploration. The robot creeps forward slowly, uses
SLAM-rendered depth to detect obstacles, and steers / stops / backs up
as needed. The current map is periodically exported as PLY for the web
dashboard to display.

Sensor fusion:
  - MASt3R-SLAM: dense depth + camera pose from a monocular RGB stream
  - Ultrasonic: forward-only proximity, unreliable on soft surfaces
  - Depth from SLAM rendering is the primary obstacle sensor
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tankbot.desktop.autonomy.frontier import FrontierPlanner, PlannerMode
from tankbot.desktop.autonomy.planning import AutonomyPlanner, Goal, GoalKind
from tankbot.desktop.autonomy.robot_client import RobotClient
from tankbot.desktop.autonomy.slam import SplatSLAM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Speeds (duty values, max 4095) — calibrated from human driving
EXPLORE_SPEED = 1800  # slightly slower in autonomy so SLAM can keep up
CRUISE_SPEED = 1850  # keep forward exploration deliberate, not twitch-fast
TURN_SPEED = 1500
INITIAL_SCAN_TURN_SPEED = 1500
INITIAL_SCAN_RIGHT_TURN_SPEED = 1700
NAV_TURN_SPEED = 1500
PLANNER_TURN_SPEED = 1700
RECOVERY_TURN_SPEED = 1500
MIN_DUTY = 1500

# Pulse-and-coast: short burst then stop to keep speed low
PULSE_DURATION = 0.20  # shorter forward pulses to give SLAM time to settle
COAST_PAUSE = 0.08  # seconds of coasting (motors off) between pulses
AVOIDANCE_BACKUP_DURATION = 0.10
RECOVERY_REVERSE_DURATION = 0.30
RECOVERY_RETREAT_DURATION = 0.35
RECOVERY_TURN_DURATION = 0.24
RECOVERY_SETTLE_DURATION = 0.25
RECOVERY_STABLE_FRAMES = 3
RECOVERY_UNDO_DELAY_FRAMES = 5
RECOVERY_RETRY_RETREAT_INTERVAL = 10
RECOVERY_MAX_RETREATS = 3
RECOVERY_ANCHOR_TURN_DELAY_FRAMES = 12
RECOVERY_ANCHOR_HEADING_TOLERANCE_DEG = 18.0
PLANNER_APPROACH_UNDO_DELAY_FRAMES = 8
NAV_KF_SETTLE_TURN_SECONDS = 1.0
NAV_KF_SETTLE_MOVE_SECONDS = 1.4
NAV_KF_SETTLE_MAX_SECONDS = 2.5
RECOVERY_KEYFRAME_HOLDOFF_FRAMES = 3
UNSTABLE_TRACKING_FRAME_LIMIT = 6
PLANNER_TURN_STALL_LIMIT = 6
PARALLAX_PROBE_TRIGGER_FRAMES = 18
PARALLAX_PROBE_RETURN_FRAMES = 8
PARALLAX_PROBE_MIN_CLEARANCE = 0.65
PARALLAX_PROBE_DURATION = 0.18

# Stuck detection
STUCK_POSITION_THRESHOLD = 0.02  # meters — less movement than this = stuck
STUCK_FRAME_LIMIT = 15  # consecutive "no movement" frames before declaring stuck

# Ultrasonic thresholds in CM — calibrated from human driving recording
US_STOP = 40  # closer than 40 cm → turn away (human never went below 46cm)
US_CLEAR = 100  # farther than 1m → full confidence
SLAM_FRONT_STOP = 0.45  # hard stop if SLAM center strip says obstacle is too close
SLAM_FRONT_SLOW = 0.80  # start slowing before the hard stop
RECOVERY_RETREAT_US_NEAR = 45
RECOVERY_RETREAT_SLAM_NEAR = 0.75

# Initial scan: rotate in place to build panoramic map before moving
INITIAL_SCAN_STEPS = 12  # number of turn pulses at startup
INITIAL_SCAN_EXTRA_STEPS = 4
INITIAL_SCAN_MIN_KEYFRAMES = 10
INITIAL_SCAN_MAX_ROUNDS = 5
INITIAL_SCAN_CORRECTION_INTERVAL = 4
STARTUP_MIN_KEYFRAMES = 1
STARTUP_SETTLE_SECONDS = 1.5
INITIAL_SCAN_TURN_DURATION = 0.24
INITIAL_SCAN_RECOVERY_REVERSE_DURATION = 0.25
NAV_TURN_DURATION = 0.26
STUCK_TURN_DURATION = 0.30

# We divide the frame into three vertical strips for steering decisions
STRIP_LEFT = (0, FRAME_WIDTH // 3)
STRIP_CENTER = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)
STRIP_RIGHT = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)

US_MAX_TRUSTED = 200

# State hysteresis
MIN_STATE_DURATION = {
    "scanning": 0.4,
    "cruising": 0.3,
    "avoiding": 0.6,
    "backing_up": 0.5,
    "stopped": 0.3,
}

# SLAM
PLY_EXPORT_INTERVAL = 5.0  # seconds between PLY exports


class State(str, Enum):
    SCANNING = "scanning"
    CRUISING = "cruising"
    AVOIDING = "avoiding"
    BACKING_UP = "backing_up"
    STOPPED = "stopped"


class ControlPhase(str, Enum):
    BOOTSTRAP = "bootstrap"
    INITIAL_SCAN = "initial_scan"
    NAVIGATING = "navigating"
    RECOVERING = "recovering"


@dataclass
class FrameContext:
    us_distance: float
    depth_strips: tuple[float, float, float]
    slam_result: object | None


# Set True to prevent motor commands regardless of --debug flag
FORCE_DEBUG = False


class VisionEngine:
    def __init__(
        self, robot_url: str = "", debug: bool = False, splat_dir: str = "", record: bool = False, calib_path: str = ""
    ) -> None:
        debug = debug or FORCE_DEBUG
        self._record = record
        self._record_file = None
        self._intent_trace_file = None
        self._intent_trace_writer = None
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._client = RobotClient(robot_url)
        self._slam = SplatSLAM(calibration_path=calib_path)
        self._latest_frame: bytes | None = None
        self._latest_telemetry: dict = {}
        self._running = False
        self._debug = debug
        self._state = State.SCANNING
        self._phase = ControlPhase.BOOTSTRAP
        self._state_since = time.monotonic()
        self._planner = AutonomyPlanner(
            Goal(
                kind=GoalKind.EXPLORE_ROOM,
                description="Explore the room while building and maintaining a map.",
            )
        )
        self._frontier_planner = FrontierPlanner()
        self._planner_snapshot = self._frontier_planner.snapshot()
        self._last_planner_mode = None
        self._last_planner_command_mode = None
        self._last_planner_command = None
        self._planner_turn_stall_count = 0
        self._planner_last_turn_target = None
        self._last_behavior = None
        self._current_motor = (0, 0)
        self._last_nonzero_motor = (0, 0)
        self._last_maneuver: tuple[int, int, float] | None = None
        self._last_forward_maneuver: tuple[int, int, float] | None = None
        self._last_issued_command: tuple[int, int, float] | None = None
        self._last_stable_pose = None
        self._recovery_retreat_count = 0
        self._us_history: list[float] = []

        # PLY export state
        self._splat_dir = splat_dir or self._default_splat_dir()
        self._ply_version = 0
        self._ply_epoch = int(time.time() * 1000)
        self._last_ply_export = 0.0
        self._send_counter = 0
        self._frame_count = 0
        self._last_nav_pose = None
        self._stuck_count = 0
        self._last_kf_frame = 0  # frame count when last keyframe was added
        self._initial_scan_done = False
        self._scan_steps_remaining = 0
        self._scan_rounds = 0
        self._scan_step_index = 0
        self._startup_ready_since: float | None = None
        self._lost_recovery_reversed = False
        self._scan_loss_recovery_reversed = False
        self._resume_initial_scan_after_recovery = False
        self._recovery_stable_frames = 0
        self._unstable_tracking_count = 0
        self._pending_probe_return = False
        self._probe_start_frame = 0
        self._last_led_payload: tuple[int, int, int, int] | None = None
        self._last_led_update = 0.0
        self._nav_settle_until = 0.0
        self._nav_waiting_for_kf = False
        # depth_scale no longer used — US for distance, SLAM for steering only

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle

    @staticmethod
    def _heading_from_pose(camera_pose) -> float:
        forward = camera_pose[:3, :3] @ [0.0, 0.0, 1.0]
        return math.atan2(float(forward[2]), float(forward[0]))

    @staticmethod
    def _default_splat_dir() -> str:
        """Default PLY output: Phoenix priv/static/assets/splat/."""
        candidates = [
            Path(__file__).resolve().parents[4] / "tankbot_web" / "priv" / "static" / "assets" / "splat",
            Path.cwd() / "tankbot_web" / "priv" / "static" / "assets" / "splat",
        ]
        for p in candidates:
            if p.parent.exists():
                p.mkdir(parents=True, exist_ok=True)
                return str(p)
        # Fallback to temp
        fallback = Path("/tmp/tankbot_splat")
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        self._slam.load()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    async def _on_frame(self, jpeg: bytes) -> None:
        self._latest_frame = jpeg

    async def _on_telemetry(self, data: dict) -> None:
        self._latest_telemetry = data

    # ------------------------------------------------------------------
    # SLAM analysis
    # ------------------------------------------------------------------
    def _analyse_frame(self, jpeg: bytes) -> dict:
        """Run SLAM on a frame — returns depth strips + pose info."""
        result = self._slam.process_frame(jpeg, time.monotonic())

        if result is None:
            # During warmup or tracking failure
            return {
                "detections": [],
                "depth_strips": (99.0, 99.0, 99.0),
                "slam_result": None,
            }

        strip_l, strip_c, strip_r = self._slam.get_strip_distances(result.depth_map)

        return {
            "detections": [],
            "depth_strips": (strip_l, strip_c, strip_r),
            "slam_result": result,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_state(self, new: State) -> None:
        if new != self._state:
            log.info("State: %s -> %s", self._state.value, new.value)
            self._state = new
            self._state_since = time.monotonic()

    def _set_phase(self, new: ControlPhase) -> None:
        if new != self._phase:
            log.info("Phase: %s -> %s", self._phase.value, new.value)
            self._phase = new
            self._log_behavior_transition()

    def _log_behavior_transition(self) -> None:
        snapshot = self._planner.snapshot(phase=self._phase.value)
        if snapshot.behavior != self._last_behavior:
            log.info(
                "Behavior: %s -> %s (goal=%s)",
                self._last_behavior.value if self._last_behavior is not None else "none",
                snapshot.behavior.value,
                snapshot.goal.kind.value,
            )
            self._last_behavior = snapshot.behavior

    async def _motor(self, left: int, right: int) -> None:
        self._current_motor = (left, right)
        if left != 0 or right != 0:
            self._last_nonzero_motor = (left, right)
        await self._client.send_motor(left, right)

    async def _stop(self) -> None:
        await self._motor(0, 0)

    async def _run_maneuver(self, left: int, right: int, duration: float, *, remember: bool = True) -> None:
        self._last_issued_command = (left, right, duration)
        if remember and (left != 0 or right != 0):
            self._last_maneuver = (left, right, duration)
            if left > 0 and right > 0:
                self._last_forward_maneuver = (left, right, duration)
        await self._motor(left, right)
        await asyncio.sleep(duration)
        await self._stop()

    async def _run_turn(self, turn_dir: int, speed: int, duration: float, *, remember: bool = True) -> None:
        """Asymmetric skid turn that resists motor stall on carpet.

        `turn_dir` > 0 rotates left, < 0 rotates right.
        """
        reverse_speed = max(MIN_DUTY, int(speed * 0.85))
        if turn_dir >= 0:
            left, right = -reverse_speed, speed
        else:
            left, right = speed, -reverse_speed
        await self._run_maneuver(left, right, duration, remember=remember)

    async def _run_strong_turn(self, turn_dir: int, speed: int, duration: float, *, remember: bool = True) -> None:
        """Stronger in-place turn for startup scan where fast coverage matters more."""
        if turn_dir >= 0:
            left, right = -speed, speed
        else:
            left, right = speed, -speed
        await self._run_maneuver(left, right, duration, remember=remember)

    def _inverse_last_maneuver(self, default_duration: float) -> tuple[int, int, float] | None:
        if self._last_maneuver is None:
            return None
        left, right, duration = self._last_maneuver
        if left == 0 and right == 0:
            return None
        return -left, -right, max(duration, default_duration)

    def _begin_nav_settle(self, duration: float) -> None:
        self._nav_settle_until = time.monotonic() + min(duration, NAV_KF_SETTLE_MAX_SECONDS)
        self._nav_waiting_for_kf = True

    def _retreat_forward_maneuver(self, default_duration: float) -> tuple[int, int, float] | None:
        if self._last_forward_maneuver is None:
            return None
        left, right, duration = self._last_forward_maneuver
        if left <= 0 or right <= 0:
            return None
        return -left, -right, max(duration, default_duration)

    def _filter_distance(self, raw: float) -> float:
        if raw <= 0 or raw > US_MAX_TRUSTED:
            valid = None
        else:
            valid = raw
        if valid is not None:
            self._us_history.append(valid)
        self._us_history = self._us_history[-3:]
        if not self._us_history:
            return float("inf")
        return min(self._us_history)

    def _maybe_export_ply(self) -> None:
        """Export PLY if enough time has passed since last export."""
        now = time.monotonic()
        if now - self._last_ply_export < PLY_EXPORT_INTERVAL:
            return
        ply_path = os.path.join(self._splat_dir, "scene.ply")
        if self._slam.export_ply(ply_path):
            self._ply_version += 1
            self._last_ply_export = now
            log.info("PLY v%d exported (%d points)", self._ply_version, self._slam.exported_point_count)

    def _status_detail(self, slam_result) -> str:
        if self._phase == ControlPhase.BOOTSTRAP:
            return "Warming up SLAM before movement"
        if self._phase == ControlPhase.INITIAL_SCAN:
            return f"Initial scan round {self._scan_rounds}, {self._scan_steps_remaining} steps remaining"
        if self._phase == ControlPhase.RECOVERING:
            return f"Recovering tracking after {getattr(self, '_lost_count', 0)} lost frames"
        if self._last_planner_command is None:
            return "Waiting for planner command"
        target = self._last_planner_command.target_cell
        target_txt = f" target={target}" if target is not None else ""
        return f"{self._last_planner_command.mode.value}: {self._last_planner_command.reason}{target_txt}"

    async def _update_led_feedback(self) -> None:
        if self._debug:
            return

        now = time.monotonic()
        if now - self._last_led_update < 0.18:
            return
        step = (self._frame_count // 2) % 8
        payload = self._led_payload(step)
        if payload != self._last_led_payload:
            r, g, b, mask = payload
            await self._client.send_led(r, g, b, mask)
            self._last_led_payload = payload
        self._last_led_update = now

    def _led_payload(self, step: int) -> tuple[int, int, int, int]:
        if self._phase == ControlPhase.BOOTSTRAP:
            pulse = 40 + (step % 4) * 25
            return (0, pulse, pulse, 0xF)
        if self._phase == ControlPhase.INITIAL_SCAN:
            masks = [0x1, 0x3, 0x6, 0xC, 0x8, 0xC, 0x6, 0x3]
            return (0, 60, 200, masks[step % len(masks)])
        if self._phase == ControlPhase.RECOVERING:
            masks = [0x5, 0xA]
            return (220, 30, 20, masks[step % len(masks)])

        mode = self._last_planner_command_mode
        if self._state == State.BACKING_UP:
            masks = [0x9, 0x6]
            return (255, 80, 0, masks[step % len(masks)])
        if self._state == State.AVOIDING:
            masks = [0x3, 0xC]
            return (220, 180, 0, masks[step % len(masks)])
        if mode == PlannerMode.TURN:
            masks = [0x1, 0x2, 0x4, 0x8]
            heading = self._last_planner_command.target_heading if self._last_planner_command else None
            if heading is not None and heading < 0:
                masks = list(reversed(masks))
            return (255, 80, 0, masks[step % len(masks)])
        if mode in {PlannerMode.APPROACH_FRONTIER, PlannerMode.FORWARD}:
            pulse = 80 + (step % 4) * 30
            return (0, pulse, 40, 0xF)
        if mode in {PlannerMode.HOLD, PlannerMode.REVISIT_ANCHOR, PlannerMode.RECOVERY_SCAN}:
            return (120, 120, 120, 0xF)
        return (0, 140, 60, 0xF)

    def _startup_ready(self, slam_result) -> bool:
        if slam_result is None or slam_result.tracking_lost or slam_result.num_keyframes < STARTUP_MIN_KEYFRAMES:
            self._startup_ready_since = None
            return False

        now = time.monotonic()
        if self._startup_ready_since is None:
            self._startup_ready_since = now
            log.info("Startup warmup: first keyframe available, settling for %.1fs", STARTUP_SETTLE_SECONDS)
            return False

        return (now - self._startup_ready_since) >= STARTUP_SETTLE_SECONDS

    async def _send_vision_status(
        self, us_distance: float, depth_strips: tuple[float, float, float], slam_result
    ) -> None:
        self._send_counter += 1
        plan = self._planner.snapshot(phase=self._phase.value)

        # Build SLAM telemetry (sent every update)
        slam_data = None
        if slam_result is not None:
            pose_flat = None
            if slam_result.pose_valid:
                # Sanitize pose — replace NaN/inf with 0 for JSON safety
                pose_flat = slam_result.camera_pose.flatten().tolist()
                pose_flat = [0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 6) for v in pose_flat]

            slam_data = {
                "tracking_quality": round(float(slam_result.tracking_quality), 3),
                "num_points": int(slam_result.num_points),
                "camera_pose": pose_flat,
                "pose_valid": bool(slam_result.pose_valid),
                "ply_epoch": self._ply_epoch,
                "ply_version": self._ply_version,
            }

        try:
            await self._client.send_vision_status(
                state=self._state.value,
                detections=[],
                distance=us_distance,
                depth_strips=depth_strips,
                slam_data=slam_data,
                autonomy_data={
                    "goal": plan.goal.kind.value,
                    "behavior": plan.behavior.value,
                    "phase": plan.phase,
                    "planner_mode": self._planner_snapshot.planner_mode,
                    "frontier_count": self._planner_snapshot.frontier_count,
                    "coverage_ratio": round(float(self._planner_snapshot.coverage_ratio), 3),
                    "planner_reason": self._planner_snapshot.reason,
                    "selected_frontier": self._planner_snapshot.selected_frontier,
                    "status_detail": self._status_detail(slam_result),
                    "lost_count": int(getattr(self, "_lost_count", 0)),
                    "scan_steps_remaining": int(self._scan_steps_remaining),
                    "scan_round": int(self._scan_rounds),
                    "planner_target_heading_deg": (
                        round(math.degrees(self._last_planner_command.target_heading), 1)
                        if self._last_planner_command is not None
                        and self._last_planner_command.target_heading is not None
                        else None
                    ),
                    "planner_target_cell": (
                        list(self._last_planner_command.target_cell)
                        if self._last_planner_command is not None and self._last_planner_command.target_cell is not None
                        else None
                    ),
                },
            )
        except Exception:
            # Log full message for debugging JSON serialization issues
            if self._send_counter % 30 == 0:
                log.exception(
                    "Failed to send vision status, slam_data keys: %s", list(slam_data.keys()) if slam_data else None
                )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _init_recording(self) -> None:
        """Start recording driving data to CSV."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"recordings/drive_{ts}.csv"
        os.makedirs("recordings", exist_ok=True)
        self._record_file = open(path, "w", newline="")
        self._record_writer = csv.writer(self._record_file)
        self._record_writer.writerow(
            [
                "timestamp",
                "frame_idx",
                "motor_left",
                "motor_right",
                "us_distance",
                "slam_strip_l",
                "slam_strip_c",
                "slam_strip_r",
                "pose_x",
                "pose_y",
                "pose_z",
                "pose_rot_deg",
                "tracking_lost",
                "num_points",
            ]
        )
        log.info("Recording to %s", path)

    def _init_intent_trace(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"recordings/intent_{ts}.csv"
        os.makedirs("recordings", exist_ok=True)
        self._intent_trace_file = open(path, "w", newline="")
        self._intent_trace_writer = csv.writer(self._intent_trace_file)
        self._intent_trace_writer.writerow(
            [
                "timestamp",
                "frame_idx",
                "phase",
                "state",
                "behavior",
                "status_detail",
                "planner_mode",
                "planner_reason",
                "planner_target_cell",
                "planner_target_heading_deg",
                "frontier_count",
                "coverage_ratio",
                "pose_valid",
                "pose_x",
                "pose_y",
                "pose_z",
                "pose_heading_deg",
                "tracking_lost",
                "tracking_stable",
                "lost_count",
                "recovery_stable_frames",
                "unstable_tracking_count",
                "scan_round",
                "scan_steps_remaining",
                "num_keyframes",
                "num_points",
                "us_distance_cm",
                "depth_left_m",
                "depth_center_m",
                "depth_right_m",
                "issued_left",
                "issued_right",
                "issued_duration_s",
                "motor_left",
                "motor_right",
            ]
        )
        log.info("Intent trace to %s", path)

    def _record_intent_trace(self, ctx: FrameContext, planner_cmd) -> None:
        if self._intent_trace_writer is None:
            return
        behavior = self._planner.snapshot(phase=self._phase.value).behavior.value
        slam_result = ctx.slam_result
        status_detail = self._status_detail(slam_result)
        target_cell = (
            f"{planner_cmd.target_cell[0]},{planner_cmd.target_cell[1]}"
            if planner_cmd is not None and planner_cmd.target_cell is not None
            else ""
        )
        target_heading = (
            f"{math.degrees(planner_cmd.target_heading):.1f}"
            if planner_cmd is not None and planner_cmd.target_heading is not None
            else ""
        )
        pose_valid = bool(slam_result.pose_valid) if slam_result is not None else False
        pose_x = pose_y = pose_z = pose_heading = ""
        if pose_valid:
            pose_x = f"{slam_result.camera_pose[0, 3]:.4f}"
            pose_y = f"{slam_result.camera_pose[1, 3]:.4f}"
            pose_z = f"{slam_result.camera_pose[2, 3]:.4f}"
            pose_heading = f"{math.degrees(self._heading_from_pose(slam_result.camera_pose)):.1f}"
        issued_left = issued_right = issued_duration = ""
        if self._last_issued_command is not None:
            issued_left = self._last_issued_command[0]
            issued_right = self._last_issued_command[1]
            issued_duration = f"{self._last_issued_command[2]:.2f}"
        self._intent_trace_writer.writerow(
            [
                f"{time.monotonic():.3f}",
                self._frame_count,
                self._phase.value,
                self._state.value,
                behavior,
                status_detail,
                planner_cmd.mode.value if planner_cmd is not None else "",
                planner_cmd.reason if planner_cmd is not None else "",
                target_cell,
                target_heading,
                self._planner_snapshot.frontier_count,
                f"{self._planner_snapshot.coverage_ratio:.4f}",
                1 if pose_valid else 0,
                pose_x,
                pose_y,
                pose_z,
                pose_heading,
                1 if (slam_result and slam_result.tracking_lost) else 0,
                1 if (slam_result and getattr(slam_result, "tracking_stable", True)) else 0,
                int(getattr(self, "_lost_count", 0)),
                int(getattr(self, "_recovery_stable_frames", 0)),
                int(getattr(self, "_unstable_tracking_count", 0)),
                int(self._scan_rounds),
                int(self._scan_steps_remaining),
                int(slam_result.num_keyframes) if slam_result is not None else 0,
                int(slam_result.num_points) if slam_result is not None else 0,
                f"{ctx.us_distance:.1f}",
                f"{ctx.depth_strips[0]:.3f}",
                f"{ctx.depth_strips[1]:.3f}",
                f"{ctx.depth_strips[2]:.3f}",
                issued_left,
                issued_right,
                issued_duration,
                self._current_motor[0],
                self._current_motor[1],
            ]
        )
        self._intent_trace_file.flush()

    def _record_frame(self, slam_result, us_distance: float, strips: tuple, motor_left: int, motor_right: int) -> None:
        """Record one frame of driving data."""
        if self._record_writer is None:
            return
        import math

        pose = slam_result.camera_pose if slam_result else None
        if pose is not None:
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            trace = pose[0, 0] + pose[1, 1] + pose[2, 2]
            rot = math.degrees(math.acos(max(-1, min(1, (trace - 1) / 2))))
        else:
            tx = ty = tz = rot = 0.0

        self._record_writer.writerow(
            [
                f"{time.monotonic():.3f}",
                self._frame_count,
                motor_left,
                motor_right,
                f"{us_distance:.1f}",
                f"{strips[0]:.3f}",
                f"{strips[1]:.3f}",
                f"{strips[2]:.3f}",
                f"{tx:.4f}",
                f"{ty:.4f}",
                f"{tz:.4f}",
                f"{rot:.1f}",
                1 if (slam_result and slam_result.tracking_lost) else 0,
                slam_result.num_points if slam_result else 0,
            ]
        )
        self._record_file.flush()

    def _build_frame_context(self, analysis: dict, us_distance: float) -> FrameContext:
        return FrameContext(
            us_distance=us_distance,
            depth_strips=analysis["depth_strips"],
            slam_result=analysis["slam_result"],
        )

    async def _handle_bootstrap_phase(self, ctx: FrameContext) -> bool:
        if self._initial_scan_done:
            self._set_phase(ControlPhase.NAVIGATING)
            return False
        if self._scan_steps_remaining > 0:
            self._set_phase(ControlPhase.INITIAL_SCAN)
            return False
        if not self._startup_ready(ctx.slam_result):
            self._set_state(State.SCANNING)
            if not self._debug:
                await self._stop()
            await asyncio.sleep(0.1)
            return True
        self._set_phase(ControlPhase.INITIAL_SCAN)
        return False

    async def _handle_initial_scan_phase(self, ctx: FrameContext) -> bool:
        if self._initial_scan_done:
            self._set_phase(ControlPhase.NAVIGATING)
            return False
        if self._debug:
            self._initial_scan_done = True
            self._set_phase(ControlPhase.NAVIGATING)
            return False

        if self._scan_steps_remaining == 0:
            self._scan_steps_remaining = INITIAL_SCAN_STEPS
            self._scan_rounds = 1
            self._scan_step_index = 0
            log.info("Starting initial scan (%d steps)", INITIAL_SCAN_STEPS)

        if ctx.slam_result is not None and ctx.slam_result.tracking_lost:
            await self._stop()
            if not self._scan_loss_recovery_reversed:
                inverse = self._inverse_last_maneuver(INITIAL_SCAN_RECOVERY_REVERSE_DURATION)
                if inverse is not None:
                    unwind_left, unwind_right, unwind_duration = inverse
                    log.info(
                        "Initial scan lost tracking — undoing last turn (%d, %d) for %.2fs",
                        unwind_left,
                        unwind_right,
                        unwind_duration,
                    )
                    await self._run_maneuver(
                        unwind_left,
                        unwind_right,
                        unwind_duration,
                        remember=False,
                    )
                    self._scan_loss_recovery_reversed = True
            else:
                log.info("Initial scan recovery failed — switching to recovery before resuming scan")
                self._resume_initial_scan_after_recovery = True
                if self._scan_steps_remaining <= 0:
                    self._scan_steps_remaining = INITIAL_SCAN_EXTRA_STEPS
                self._scan_loss_recovery_reversed = False
                self._set_phase(ControlPhase.RECOVERING)
            await asyncio.sleep(0.4)
            return True

        self._set_state(State.SCANNING)
        turn_dir = -1 if ((self._scan_step_index + 1) % INITIAL_SCAN_CORRECTION_INTERVAL == 0) else 1
        turn_speed = INITIAL_SCAN_TURN_SPEED if turn_dir >= 0 else INITIAL_SCAN_RIGHT_TURN_SPEED
        await self._run_strong_turn(turn_dir, turn_speed, INITIAL_SCAN_TURN_DURATION)
        await asyncio.sleep(0.35)
        self._scan_steps_remaining -= 1
        self._scan_step_index += 1
        self._scan_loss_recovery_reversed = False

        if self._scan_steps_remaining <= 0:
            num_keyframes = ctx.slam_result.num_keyframes if ctx.slam_result is not None else 0
            if num_keyframes < INITIAL_SCAN_MIN_KEYFRAMES and self._scan_rounds < INITIAL_SCAN_MAX_ROUNDS:
                self._scan_steps_remaining = INITIAL_SCAN_EXTRA_STEPS
                self._scan_rounds += 1
                self._scan_step_index = 0
                log.info(
                    "Initial scan produced only %d keyframes — extending scan by %d steps (round %d/%d)",
                    num_keyframes,
                    INITIAL_SCAN_EXTRA_STEPS,
                    self._scan_rounds,
                    INITIAL_SCAN_MAX_ROUNDS,
                )
            else:
                self._initial_scan_done = True
                self._set_phase(ControlPhase.NAVIGATING)
                log.info(
                    "Initial scan complete — starting exploration with %d keyframes",
                    num_keyframes,
                )
        return True

    async def _handle_recovery_phase(self, ctx: FrameContext) -> bool:
        slam_result = ctx.slam_result
        if slam_result is None:
            await asyncio.sleep(0.1)
            return True
        if not slam_result.tracking_lost:
            self._recovery_stable_frames += 1
            if self._recovery_stable_frames < RECOVERY_STABLE_FRAMES:
                self._slam.suppress_keyframes(RECOVERY_KEYFRAME_HOLDOFF_FRAMES)
                log.info(
                    "Recovery probation: %d/%d tracked frames",
                    self._recovery_stable_frames,
                    RECOVERY_STABLE_FRAMES,
                )
                if not self._debug:
                    await self._stop()
                await asyncio.sleep(RECOVERY_SETTLE_DURATION)
                return True

            if getattr(self, "_lost_count", 0) > 0:
                log.info(
                    "Tracking recovered after %d lost frames (%d stable frames)",
                    self._lost_count,
                    self._recovery_stable_frames,
                )
            self._lost_count = 0
            self._lost_recovery_reversed = False
            self._recovery_stable_frames = 0
            self._recovery_retreat_count = 0
            self._pending_probe_return = False
            if self._resume_initial_scan_after_recovery and not self._initial_scan_done:
                if slam_result.num_keyframes < INITIAL_SCAN_MIN_KEYFRAMES or self._scan_steps_remaining > 0:
                    if self._scan_steps_remaining <= 0:
                        self._scan_steps_remaining = INITIAL_SCAN_EXTRA_STEPS
                    self._resume_initial_scan_after_recovery = False
                    log.info(
                        "Recovery complete — resuming initial scan (%d steps remaining, %d keyframes)",
                        self._scan_steps_remaining,
                        slam_result.num_keyframes,
                    )
                    self._set_phase(ControlPhase.INITIAL_SCAN)
                    return False
                self._resume_initial_scan_after_recovery = False
            self._set_phase(ControlPhase.NAVIGATING)
            return False

        if not hasattr(self, "_lost_count"):
            self._lost_count = 0
        self._lost_count += 1
        self._recovery_stable_frames = 0
        self._pending_probe_return = False
        self._slam.suppress_keyframes(RECOVERY_KEYFRAME_HOLDOFF_FRAMES)

        if not self._debug:
            if self._lost_count == 1:
                log.info("Tracking lost — stopping")
                self._lost_recovery_reversed = False
                self._recovery_retreat_count = 0
                await self._stop()
            elif not self._lost_recovery_reversed and self._lost_count >= (
                PLANNER_APPROACH_UNDO_DELAY_FRAMES
                if self._last_planner_command_mode == PlannerMode.APPROACH_FRONTIER
                else RECOVERY_UNDO_DELAY_FRAMES
            ):
                near_wall = (0 < ctx.us_distance < RECOVERY_RETREAT_US_NEAR) or (
                    0 < ctx.depth_strips[1] < RECOVERY_RETREAT_SLAM_NEAR
                )
                inverse = (
                    self._retreat_forward_maneuver(RECOVERY_RETREAT_DURATION)
                    if near_wall and self._recovery_retreat_count < RECOVERY_MAX_RETREATS
                    else None
                )
                if inverse is not None:
                    reverse_left, reverse_right, reverse_duration = inverse
                    log.info(
                        "Tracking lost near wall (%d frames) — retreating along last forward path (%d, %d) for %.2fs",
                        self._lost_count,
                        reverse_left,
                        reverse_right,
                        reverse_duration,
                    )
                    self._set_state(State.BACKING_UP)
                    await self._run_maneuver(
                        reverse_left,
                        reverse_right,
                        reverse_duration,
                        remember=False,
                    )
                    self._recovery_retreat_count += 1
                    self._lost_recovery_reversed = True
                else:
                    inverse = self._inverse_last_maneuver(RECOVERY_REVERSE_DURATION)
                    if inverse is not None:
                        reverse_left, reverse_right, reverse_duration = inverse
                        log.info(
                            "Tracking lost (%d frames) — undoing last maneuver (%d, %d) for %.2fs",
                            self._lost_count,
                            reverse_left,
                            reverse_right,
                            reverse_duration,
                        )
                        self._set_state(State.BACKING_UP)
                        await self._run_maneuver(
                            reverse_left,
                            reverse_right,
                            reverse_duration,
                            remember=False,
                        )
                        self._lost_recovery_reversed = True
            elif (
                self._lost_count % RECOVERY_RETRY_RETREAT_INTERVAL == 0
                and self._recovery_retreat_count < RECOVERY_MAX_RETREATS
            ):
                inverse = self._retreat_forward_maneuver(RECOVERY_RETREAT_DURATION)
                if inverse is not None:
                    reverse_left, reverse_right, reverse_duration = inverse
                    log.info(
                        "Tracking still lost (%d frames) — widening context with another retreat (%d, %d) for %.2fs",
                        self._lost_count,
                        reverse_left,
                        reverse_right,
                        reverse_duration,
                    )
                    self._set_state(State.BACKING_UP)
                    await self._run_maneuver(
                        reverse_left,
                        reverse_right,
                        reverse_duration,
                        remember=False,
                    )
                    self._recovery_retreat_count += 1
            elif self._lost_count % RECOVERY_ANCHOR_TURN_DELAY_FRAMES == 0:
                log.info("Tracking lost (%d frames) — turning toward last stable anchor", self._lost_count)
                self._set_state(State.SCANNING)
                turn_dir = -1 if self._last_nonzero_motor[0] > self._last_nonzero_motor[1] else 1
                if self._last_stable_pose is not None and slam_result.pose_valid:
                    anchor_heading = self._heading_from_pose(self._last_stable_pose)
                    current_heading = self._heading_from_pose(slam_result.camera_pose)
                    heading_error = self._wrap_angle(anchor_heading - current_heading)
                    if abs(math.degrees(heading_error)) > RECOVERY_ANCHOR_HEADING_TOLERANCE_DEG:
                        turn_dir = 1 if heading_error >= 0 else -1
                await self._run_turn(turn_dir, RECOVERY_TURN_SPEED, RECOVERY_TURN_DURATION)

        await asyncio.sleep(RECOVERY_SETTLE_DURATION)
        return True

    async def _handle_navigation_phase(self, ctx: FrameContext) -> bool:
        slam_result = ctx.slam_result
        if slam_result is None:
            self._set_state(State.SCANNING)
            if not self._debug:
                await self._stop()
            await asyncio.sleep(0.1)
            return True

        if slam_result.tracking_lost:
            self._unstable_tracking_count = 0
            self._set_phase(ControlPhase.RECOVERING)
            return False

        if getattr(slam_result, "tracking_stable", True) and slam_result.pose_valid:
            self._last_stable_pose = slam_result.camera_pose.copy()

        self._lost_count = 0
        self._lost_recovery_reversed = False

        if not getattr(slam_result, "tracking_stable", True):
            self._unstable_tracking_count += 1
        else:
            self._unstable_tracking_count = 0

        if self._unstable_tracking_count >= UNSTABLE_TRACKING_FRAME_LIMIT:
            log.info(
                "Tracking unstable for %d frames — switching to recovery",
                self._unstable_tracking_count,
            )
            self._unstable_tracking_count = 0
            self._set_phase(ControlPhase.RECOVERING)
            return False

        strip_l, strip_c, strip_r = ctx.depth_strips
        us_distance = ctx.us_distance
        total_strip = strip_l + strip_c + strip_r

        self._frontier_planner.update_from_frame(
            slam_result.camera_pose,
            slam_result.planning_points,
            pose_valid=slam_result.pose_valid,
            tracking_lost=slam_result.tracking_lost,
        )
        planner_cmd = self._frontier_planner.command_for_state(
            slam_result.camera_pose,
            pose_valid=slam_result.pose_valid,
            tracking_lost=slam_result.tracking_lost,
            tracking_stable=getattr(slam_result, "tracking_stable", True),
        )
        self._planner_snapshot = self._frontier_planner.snapshot()
        if planner_cmd.mode != self._last_planner_mode or self._frame_count % 15 == 0:
            log.info(
                "Planner: mode=%s reason=%s frontiers=%d coverage=%.2f target=%s heading=%s conf=%.2f",
                planner_cmd.mode.value,
                planner_cmd.reason,
                self._planner_snapshot.frontier_count,
                self._planner_snapshot.coverage_ratio,
                planner_cmd.target_cell,
                f"{math.degrees(planner_cmd.target_heading):.1f}deg" if planner_cmd.target_heading is not None else "—",
                planner_cmd.confidence,
            )
            self._last_planner_mode = planner_cmd.mode
        self._last_planner_command_mode = planner_cmd.mode
        self._last_planner_command = planner_cmd

        if self._debug:
            if self._frame_count % 30 == 0:
                log.info(
                    "US=%.1fcm | SLAM L/C/R=%.1f/%.1f/%.1f | points=%d | planner=%s frontiers=%d coverage=%.2f",
                    us_distance,
                    strip_l,
                    strip_c,
                    strip_r,
                    slam_result.num_points,
                    self._planner_snapshot.planner_mode,
                    self._planner_snapshot.frontier_count,
                    self._planner_snapshot.coverage_ratio,
                )
            await asyncio.sleep(0.1)
            return True

        self._record_intent_trace(ctx, planner_cmd)

        import numpy as np

        front_clearance_m = strip_c
        if (0 < us_distance < US_STOP) or (front_clearance_m > 0 and front_clearance_m < SLAM_FRONT_STOP):
            self._set_state(State.SCANNING)
            await self._stop()
            if strip_l > strip_r:
                await self._run_turn(1, NAV_TURN_SPEED, NAV_TURN_DURATION)
            else:
                await self._run_turn(-1, NAV_TURN_SPEED, NAV_TURN_DURATION)
            await asyncio.sleep(0.4)
            return True

        if us_distance > 0:
            clearance = max(0.0, min(1.0, (us_distance - US_STOP) / (US_CLEAR - US_STOP)))
        else:
            clearance = 0.7
        if front_clearance_m > 0:
            slam_clearance = max(
                0.0, min(1.0, (front_clearance_m - SLAM_FRONT_STOP) / (SLAM_FRONT_SLOW - SLAM_FRONT_STOP))
            )
            clearance = min(clearance, slam_clearance)

        if slam_result.new_keyframe:
            self._last_kf_frame = self._frame_count
            self._pending_probe_return = False
            self._nav_waiting_for_kf = False
            self._nav_settle_until = 0.0
        frames_since_kf = self._frame_count - self._last_kf_frame
        familiarity = min(1.0, frames_since_kf / 50.0)
        top_speed = int(EXPLORE_SPEED + (CRUISE_SPEED - EXPLORE_SPEED) * familiarity)

        if self._nav_waiting_for_kf:
            if time.monotonic() < self._nav_settle_until:
                self._set_state(State.STOPPED)
                await self._stop()
                await asyncio.sleep(COAST_PAUSE)
                return True
            self._nav_waiting_for_kf = False

        if planner_cmd.mode in {PlannerMode.HOLD, PlannerMode.REVISIT_ANCHOR, PlannerMode.RECOVERY_SCAN}:
            self._set_state(State.STOPPED)
            await self._stop()
            await asyncio.sleep(COAST_PAUSE)
            return True

        target_heading = (
            planner_cmd.target_heading
            if planner_cmd.target_heading is not None
            else self._heading_from_pose(slam_result.camera_pose)
        )
        heading_error = self._wrap_angle(target_heading - self._heading_from_pose(slam_result.camera_pose))

        if planner_cmd.mode == PlannerMode.TURN:
            turn_target = planner_cmd.target_cell
            if turn_target is not None and turn_target == self._planner_last_turn_target:
                self._planner_turn_stall_count += 1
            else:
                self._planner_turn_stall_count = 1
                self._planner_last_turn_target = turn_target

            if self._planner_turn_stall_count >= PLANNER_TURN_STALL_LIMIT:
                log.info(
                    "Planner turn stalled on target %s — deferring target and replanning",
                    turn_target,
                )
                self._frontier_planner.defer_target(turn_target)
                self._planner_turn_stall_count = 0
                self._planner_last_turn_target = None
                await self._stop()
                await asyncio.sleep(COAST_PAUSE)
                return True

            self._stuck_count = 0
            self._last_nav_pose = None
            self._set_state(State.SCANNING)
            turn_dir = 1 if heading_error >= 0 else -1
            turn_duration = min(0.40, max(0.20, abs(heading_error) / math.radians(90.0) * NAV_TURN_DURATION))
            await self._run_strong_turn(turn_dir, PLANNER_TURN_SPEED, turn_duration)
            self._begin_nav_settle(NAV_KF_SETTLE_TURN_SECONDS)
            await asyncio.sleep(COAST_PAUSE)
            return True

        self._planner_turn_stall_count = 0
        self._planner_last_turn_target = None

        bias = (strip_r - strip_l) / total_strip if total_strip > 0.01 else 0.0
        heading_steer = max(-0.8, min(0.8, heading_error / math.radians(55.0)))
        reactive_steer = bias * (0.15 + 0.25 * (1.0 - clearance))
        steer = max(-0.85, min(0.85, (0.7 * heading_steer) + (0.3 * reactive_steer)))

        base_speed = int(MIN_DUTY + (top_speed - MIN_DUTY) * clearance)
        left_speed = max(MIN_DUTY, min(int(base_speed * (1 + steer)), CRUISE_SPEED))
        right_speed = max(MIN_DUTY, min(int(base_speed * (1 - steer)), CRUISE_SPEED))

        if clearance < 0.3:
            self._set_state(State.AVOIDING)
        else:
            self._set_state(State.CRUISING)

        cur_pos = slam_result.camera_pose[:3, 3]
        if self._last_nav_pose is not None:
            movement = np.linalg.norm(cur_pos - self._last_nav_pose)
            if movement < STUCK_POSITION_THRESHOLD:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_nav_pose = cur_pos
        else:
            self._last_nav_pose = cur_pos

        if self._stuck_count >= STUCK_FRAME_LIMIT:
            log.info("Stuck detected — turning")
            self._set_state(State.SCANNING)
            if strip_l > strip_r:
                await self._run_turn(1, NAV_TURN_SPEED, STUCK_TURN_DURATION)
            else:
                await self._run_turn(-1, NAV_TURN_SPEED, STUCK_TURN_DURATION)
            self._stuck_count = 0
            self._last_nav_pose = None
            await asyncio.sleep(0.4)
            return True

        pulse = PULSE_DURATION + 0.06 * familiarity
        if planner_cmd.mode == PlannerMode.APPROACH_FRONTIER:
            pulse += 0.03
        await self._run_maneuver(left_speed, right_speed, pulse)
        self._begin_nav_settle(NAV_KF_SETTLE_MOVE_SECONDS)
        await asyncio.sleep(COAST_PAUSE)
        return True

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------
    async def _decision_loop(self) -> None:
        """Main low-level autonomy FSM.

        This layer owns immediate control states only. Future mission/behavior
        logic (explore room, go to target, find object) should sit above it.
        """
        while self._running:
            frame = self._latest_frame
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            self._latest_frame = None

            loop = asyncio.get_running_loop()
            analysis = await loop.run_in_executor(None, self._analyse_frame, frame)

            raw_depth_strips = analysis["depth_strips"]
            self._frame_count += 1

            raw_us = self._latest_telemetry.get("distance", -1)
            us_distance = self._filter_distance(raw_us)
            if us_distance == float("inf"):
                us_distance = -1.0
            ctx = self._build_frame_context(analysis, us_distance)
            slam_result = ctx.slam_result

            # SLAM depth strips are relative (no calibration needed for steering)
            strip_l, strip_c, strip_r = raw_depth_strips
            depth_strips = raw_depth_strips

            # Record driving data (captures YOUR motor commands from the UI)
            if self._record and self._record_writer:
                motor = self._latest_telemetry.get("motor", {})
                self._record_frame(
                    slam_result,
                    us_distance,
                    raw_depth_strips,
                    motor.get("left", 0),
                    motor.get("right", 0),
                )

            # Export PLY periodically (in executor since it touches disk)
            if slam_result is not None:
                await loop.run_in_executor(None, self._maybe_export_ply)

            await self._update_led_feedback()
            await self._send_vision_status(us_distance, depth_strips, slam_result)

            if self._phase == ControlPhase.BOOTSTRAP:
                handled = await self._handle_bootstrap_phase(ctx)
            elif self._phase == ControlPhase.INITIAL_SCAN:
                handled = await self._handle_initial_scan_phase(ctx)
            elif self._phase == ControlPhase.RECOVERING:
                handled = await self._handle_recovery_phase(ctx)
            else:
                handled = await self._handle_navigation_phase(ctx)

            if handled:
                continue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._running = True
        self._load_models()

        if self._record:
            self._init_recording()
        self._init_intent_trace()

        self._log_behavior_transition()

        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(self._on_telemetry)

        await self._client.connect()
        if self._debug:
            log.info("Vision engine running — DEBUG MODE (no motor commands)")
        else:
            log.info("Vision engine running — MASt3R-SLAM autonomous mode active")

        try:
            await asyncio.gather(
                self._client.listen(),
                self._decision_loop(),
            )
        except KeyboardInterrupt:
            pass
        finally:
            if not self._debug:
                await self._client.send_stop()
                await self._client.send_led_off()
            if self._record_file:
                self._record_file.close()
                log.info("Recording saved")
            if self._intent_trace_file:
                self._intent_trace_file.close()
                log.info("Intent trace saved")
            await self._client.close()
            log.info("Vision engine stopped")


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tank robot autonomous vision (MASt3R-SLAM)")
    parser.add_argument("--robot", default="", help="Robot WebSocket URL (or set ROBOT_URL env var)")
    parser.add_argument("--splat-dir", default="", help="Directory for PLY export (default: Phoenix static assets)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: run SLAM but send no motor commands")
    parser.add_argument(
        "--record", action="store_true", help="Record driving data (use with --debug while driving manually)"
    )
    parser.add_argument(
        "--calib",
        default="",
        help="Path to MASt3R-SLAM intrinsics YAML (or set ROBOT_CAMERA_CALIB env var)",
    )
    args = parser.parse_args()

    engine = VisionEngine(
        robot_url=args.robot,
        debug=args.debug,
        splat_dir=args.splat_dir,
        record=args.record,
        calib_path=args.calib,
    )
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
