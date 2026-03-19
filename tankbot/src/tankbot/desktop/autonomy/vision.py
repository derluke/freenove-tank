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
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
EXPLORE_SPEED = 1500     # cautious speed in unknown territory
CRUISE_SPEED = 1900      # confident speed in mapped territory
TURN_SPEED = 1200
INITIAL_SCAN_TURN_SPEED = 1500
NAV_TURN_SPEED = 1350
RECOVERY_TURN_SPEED = 1300
MIN_DUTY = 1200

# Pulse-and-coast: short burst then stop to keep speed low
PULSE_DURATION = 0.20   # seconds of motor power per step
COAST_PAUSE = 0.08      # seconds of coasting (motors off) between pulses
AVOIDANCE_BACKUP_DURATION = 0.10
RECOVERY_REVERSE_DURATION = 0.30
RECOVERY_TURN_DURATION = 0.20
RECOVERY_SETTLE_DURATION = 0.25
RECOVERY_STABLE_FRAMES = 3
RECOVERY_KEYFRAME_HOLDOFF_FRAMES = 8

# Stuck detection
STUCK_POSITION_THRESHOLD = 0.02  # meters — less movement than this = stuck
STUCK_FRAME_LIMIT = 15           # consecutive "no movement" frames before declaring stuck

# Ultrasonic thresholds in CM — calibrated from human driving recording
US_STOP = 40              # closer than 40 cm → turn away (human never went below 46cm)
US_CLEAR = 100            # farther than 1m → full confidence

# Initial scan: rotate in place to build panoramic map before moving
INITIAL_SCAN_STEPS = 12   # number of turn pulses at startup
INITIAL_SCAN_EXTRA_STEPS = 4
INITIAL_SCAN_MIN_KEYFRAMES = 10
INITIAL_SCAN_MAX_ROUNDS = 5
STARTUP_MIN_KEYFRAMES = 1
STARTUP_SETTLE_SECONDS = 1.5
INITIAL_SCAN_TURN_DURATION = 0.20
INITIAL_SCAN_RECOVERY_REVERSE_DURATION = 0.25
NAV_TURN_DURATION = 0.22
STUCK_TURN_DURATION = 0.25

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
    def __init__(self, robot_url: str = "", debug: bool = False,
                 splat_dir: str = "", record: bool = False,
                 calib_path: str = "") -> None:
        debug = debug or FORCE_DEBUG
        self._record = record
        self._record_file = None
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
        self._last_behavior = None
        self._current_motor = (0, 0)
        self._last_nonzero_motor = (0, 0)
        self._last_maneuver: tuple[int, int, float] | None = None
        self._us_history: list[float] = []

        # PLY export state
        self._splat_dir = splat_dir or self._default_splat_dir()
        self._ply_version = 0
        self._last_ply_export = 0.0
        self._send_counter = 0
        self._frame_count = 0
        self._last_nav_pose = None
        self._stuck_count = 0
        self._last_kf_frame = 0    # frame count when last keyframe was added
        self._initial_scan_done = False
        self._scan_steps_remaining = 0
        self._scan_rounds = 0
        self._startup_ready_since: float | None = None
        self._lost_recovery_reversed = False
        self._scan_loss_recovery_reversed = False
        self._resume_initial_scan_after_recovery = False
        self._recovery_stable_frames = 0
        # depth_scale no longer used — US for distance, SLAM for steering only

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
        if remember and (left != 0 or right != 0):
            self._last_maneuver = (left, right, duration)
        await self._motor(left, right)
        await asyncio.sleep(duration)
        await self._stop()

    async def _run_turn(self, turn_dir: int, speed: int, duration: float, *, remember: bool = True) -> None:
        """Pivot turn to avoid stalling on carpet/static friction.

        `turn_dir` > 0 rotates left, < 0 rotates right.
        """
        if turn_dir >= 0:
            left, right = 0, speed
        else:
            left, right = speed, 0
        await self._run_maneuver(left, right, duration, remember=remember)

    def _inverse_last_maneuver(self, default_duration: float) -> tuple[int, int, float] | None:
        if self._last_maneuver is None:
            return None
        left, right, duration = self._last_maneuver
        if left == 0 and right == 0:
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

    def _startup_ready(self, slam_result) -> bool:
        if (
            slam_result is None
            or slam_result.tracking_lost
            or slam_result.num_keyframes < STARTUP_MIN_KEYFRAMES
        ):
            self._startup_ready_since = None
            return False

        now = time.monotonic()
        if self._startup_ready_since is None:
            self._startup_ready_since = now
            log.info("Startup warmup: first keyframe available, settling for %.1fs", STARTUP_SETTLE_SECONDS)
            return False

        return (now - self._startup_ready_since) >= STARTUP_SETTLE_SECONDS

    async def _send_vision_status(self, us_distance: float,
                                  depth_strips: tuple[float, float, float],
                                  slam_result) -> None:
        self._send_counter += 1
        plan = self._planner.snapshot(phase=self._phase.value)

        # Build SLAM telemetry (sent every update)
        slam_data = None
        if slam_result is not None:
            pose_flat = None
            if slam_result.pose_valid:
                # Sanitize pose — replace NaN/inf with 0 for JSON safety
                pose_flat = slam_result.camera_pose.flatten().tolist()
                import math
                pose_flat = [0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 6) for v in pose_flat]

            slam_data = {
                "tracking_quality": round(float(slam_result.tracking_quality), 3),
                "num_points": int(slam_result.num_points),
                "camera_pose": pose_flat,
                "pose_valid": bool(slam_result.pose_valid),
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
                },
            )
        except Exception:
            # Log full message for debugging JSON serialization issues
            if self._send_counter % 30 == 0:
                log.exception("Failed to send vision status, slam_data keys: %s",
                              list(slam_data.keys()) if slam_data else None)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _init_recording(self) -> None:
        """Start recording driving data to CSV."""
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"recordings/drive_{ts}.csv"
        os.makedirs("recordings", exist_ok=True)
        self._record_file = open(path, "w", newline="")
        self._record_writer = csv.writer(self._record_file)
        self._record_writer.writerow([
            "timestamp", "frame_idx",
            "motor_left", "motor_right",
            "us_distance",
            "slam_strip_l", "slam_strip_c", "slam_strip_r",
            "pose_x", "pose_y", "pose_z", "pose_rot_deg",
            "tracking_lost", "num_points",
        ])
        log.info("Recording to %s", path)

    def _record_frame(self, slam_result, us_distance: float,
                      strips: tuple, motor_left: int, motor_right: int) -> None:
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

        self._record_writer.writerow([
            f"{time.monotonic():.3f}", self._frame_count,
            motor_left, motor_right,
            f"{us_distance:.1f}",
            f"{strips[0]:.3f}", f"{strips[1]:.3f}", f"{strips[2]:.3f}",
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}", f"{rot:.1f}",
            1 if (slam_result and slam_result.tracking_lost) else 0,
            slam_result.num_points if slam_result else 0,
        ])
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
            log.info("Starting initial scan (%d steps)", INITIAL_SCAN_STEPS)

        if ctx.slam_result is not None and ctx.slam_result.tracking_lost:
            await self._stop()
            if not self._scan_loss_recovery_reversed:
                inverse = self._inverse_last_maneuver(INITIAL_SCAN_RECOVERY_REVERSE_DURATION)
                if inverse is not None:
                    unwind_left, unwind_right, unwind_duration = inverse
                    log.info(
                        "Initial scan lost tracking — undoing last turn (%d, %d) for %.2fs",
                        unwind_left, unwind_right, unwind_duration,
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
        turn_dir = 1
        await self._run_turn(turn_dir, INITIAL_SCAN_TURN_SPEED, INITIAL_SCAN_TURN_DURATION)
        await asyncio.sleep(0.35)
        self._scan_steps_remaining -= 1
        self._scan_loss_recovery_reversed = False

        if self._scan_steps_remaining <= 0:
            num_keyframes = ctx.slam_result.num_keyframes if ctx.slam_result is not None else 0
            if (
                num_keyframes < INITIAL_SCAN_MIN_KEYFRAMES
                and self._scan_rounds < INITIAL_SCAN_MAX_ROUNDS
            ):
                self._scan_steps_remaining = INITIAL_SCAN_EXTRA_STEPS
                self._scan_rounds += 1
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
            self._slam.suppress_keyframes(RECOVERY_KEYFRAME_HOLDOFF_FRAMES)
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
        self._slam.suppress_keyframes(RECOVERY_KEYFRAME_HOLDOFF_FRAMES)

        if not self._debug:
            if self._lost_count == 1:
                log.info("Tracking lost — stopping")
                self._lost_recovery_reversed = False
                await self._stop()
            elif not self._lost_recovery_reversed and self._lost_count >= 3:
                inverse = self._inverse_last_maneuver(RECOVERY_REVERSE_DURATION)
                if inverse is not None:
                    reverse_left, reverse_right, reverse_duration = inverse
                    log.info(
                        "Tracking lost (%d frames) — undoing last maneuver (%d, %d) for %.2fs",
                        self._lost_count, reverse_left, reverse_right, reverse_duration,
                    )
                    self._set_state(State.BACKING_UP)
                    await self._run_maneuver(
                        reverse_left,
                        reverse_right,
                        reverse_duration,
                        remember=False,
                    )
                    self._lost_recovery_reversed = True
            elif self._lost_count % 15 == 0:
                log.info("Tracking lost (%d frames) — turning to re-acquire", self._lost_count)
                self._set_state(State.SCANNING)
                turn_dir = -1 if self._last_nonzero_motor[0] > self._last_nonzero_motor[1] else 1
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
            self._set_phase(ControlPhase.RECOVERING)
            return False

        self._lost_count = 0
        self._lost_recovery_reversed = False

        strip_l, strip_c, strip_r = ctx.depth_strips
        us_distance = ctx.us_distance
        total_strip = strip_l + strip_c + strip_r

        if self._debug:
            if self._frame_count % 30 == 0:
                log.info(
                    "US=%.1fcm | SLAM L/C/R=%.1f/%.1f/%.1f | points=%d",
                    us_distance, strip_l, strip_c, strip_r, slam_result.num_points,
                )
            await asyncio.sleep(0.1)
            return True

        import numpy as np

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

        if 0 < us_distance < US_STOP:
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

        bias = (strip_r - strip_l) / total_strip if total_strip > 0.01 else 0.0
        steer_strength = 0.3 + 0.5 * (1.0 - clearance)
        steer = bias * steer_strength

        if slam_result.new_keyframe:
            self._last_kf_frame = self._frame_count
        frames_since_kf = self._frame_count - self._last_kf_frame
        familiarity = min(1.0, frames_since_kf / 50.0)
        top_speed = int(EXPLORE_SPEED + (CRUISE_SPEED - EXPLORE_SPEED) * familiarity)

        base_speed = int(MIN_DUTY + (top_speed - MIN_DUTY) * clearance)
        left_speed = max(MIN_DUTY, min(int(base_speed * (1 + steer)), CRUISE_SPEED))
        right_speed = max(MIN_DUTY, min(int(base_speed * (1 - steer)), CRUISE_SPEED))

        if clearance < 0.3:
            self._set_state(State.AVOIDING)
        else:
            self._set_state(State.CRUISING)

        pulse = PULSE_DURATION + 0.10 * familiarity
        await self._run_maneuver(left_speed, right_speed, pulse)
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
                    slam_result, us_distance, raw_depth_strips,
                    motor.get("left", 0), motor.get("right", 0),
                )

            # Export PLY periodically (in executor since it touches disk)
            if slam_result is not None:
                await loop.run_in_executor(None, self._maybe_export_ply)

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
            if self._record_file:
                self._record_file.close()
                log.info("Recording saved")
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
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: run SLAM but send no motor commands")
    parser.add_argument("--record", action="store_true",
                        help="Record driving data (use with --debug while driving manually)")
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
