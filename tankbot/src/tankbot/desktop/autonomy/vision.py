"""Vision pipeline — runs 3D Gaussian Splatting SLAM on robot camera frames.

Connects to the robot via WebSocket, receives frames, runs MonoGS SLAM
for real-time dense mapping and depth estimation, and sends motor commands
back to navigate autonomously.

Behaviour: cautious exploration.  The robot creeps forward slowly, uses
SLAM-rendered depth to detect obstacles, and steers / stops / backs up
as needed.  The 3D Gaussian map is periodically exported as PLY for the
web dashboard to display.

Sensor fusion:
  - 3DGS SLAM (MonoGS): metric depth + camera pose from a single RGB stream
  - Ultrasonic: forward-only proximity, unreliable on soft surfaces
  - Depth from SLAM rendering is the primary obstacle sensor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import Enum
from pathlib import Path

from tankbot.desktop.autonomy.robot_client import RobotClient
from tankbot.desktop.autonomy.slam import SplatSLAM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Speeds (duty values, max 4095) — calibrated from human driving (2000 forward)
EXPLORE_SPEED = 1500     # cautious speed in unknown territory
CRUISE_SPEED = 2000      # confident speed in mapped territory
SLOW_SPEED = 1100
TURN_SPEED = 1500
BACKUP_SPEED = 1200
MIN_DUTY = 1000

# Pulse-and-coast: short burst then stop to keep speed low
PULSE_DURATION = 0.25   # seconds of motor power per step
COAST_PAUSE = 0.20      # seconds of coasting (motors off) between pulses

# Stuck detection
STUCK_POSITION_THRESHOLD = 0.02  # meters — less movement than this = stuck
STUCK_FRAME_LIMIT = 15           # consecutive "no movement" frames before declaring stuck

# Ultrasonic thresholds in CM — calibrated from human driving recording
US_STOP = 40              # closer than 40 cm → turn away (human never went below 46cm)
US_SLOW = 60              # closer than 60 cm → slow down and steer
US_CLEAR = 100            # farther than 1m → full confidence

# Initial scan: rotate in place to build panoramic map before moving
INITIAL_SCAN_STEPS = 8    # number of turn pulses at startup

# We divide the frame into three vertical strips for steering decisions
STRIP_LEFT = (0, FRAME_WIDTH // 3)
STRIP_CENTER = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)
STRIP_RIGHT = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)

US_MAX_TRUSTED = 200

# Timing (seconds)
SCAN_PAUSE = 0.6

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
TRACKING_LOSS_THRESHOLD = 0.15  # below this → stop (quality is 0..1)
TRACKING_BOOTSTRAP_FRAMES = 100  # don't gate on quality during bootstrap


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class State(str, Enum):
    SCANNING = "scanning"
    CRUISING = "cruising"
    AVOIDING = "avoiding"
    BACKING_UP = "backing_up"
    STOPPED = "stopped"


# Set True to prevent motor commands regardless of --debug flag
FORCE_DEBUG = False


class VisionEngine:
    def __init__(self, robot_url: str = "", debug: bool = False,
                 splat_dir: str = "", record: bool = False) -> None:
        debug = debug or FORCE_DEBUG
        self._record = record
        self._record_file = None
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._client = RobotClient(robot_url)
        self._slam = SplatSLAM()
        self._latest_frame: bytes | None = None
        self._latest_telemetry: dict = {}
        self._running = False
        self._debug = debug
        self._state = State.SCANNING
        self._state_since = time.monotonic()
        self._last_turn_dir: int = 1
        self._current_motor = (0, 0)
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
        self._kf_rate = 0.0        # keyframes per frame (rolling)
        self._initial_scan_done = False
        self._scan_steps_remaining = 0
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
                "center_dist": 99.0,
                "slam_result": None,
            }

        strip_l, strip_c, strip_r = self._slam.get_strip_distances(result.depth_map)
        center_dist = self._slam.get_center_distance(result.depth_map)

        return {
            "detections": [],
            "depth_strips": (strip_l, strip_c, strip_r),
            "center_dist": center_dist,
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

    def _can_leave_state(self) -> bool:
        min_dur = MIN_STATE_DURATION.get(self._state.value, 0)
        return (time.monotonic() - self._state_since) >= min_dur

    async def _motor(self, left: int, right: int) -> None:
        self._current_motor = (left, right)
        await self._client.send_motor(left, right)

    async def _stop(self) -> None:
        await self._motor(0, 0)

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
            log.info("PLY v%d exported (%d Gaussians)", self._ply_version, self._slam.num_gaussians)

    async def _send_vision_status(self, us_distance: float,
                                  depth_strips: tuple[float, float, float],
                                  slam_result) -> None:
        self._send_counter += 1

        # Build SLAM telemetry (sent every update)
        slam_data = None
        if slam_result is not None:
            # Sanitize pose — replace NaN/inf with 0 for JSON safety
            pose_flat = slam_result.camera_pose.flatten().tolist()
            import math
            pose_flat = [0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 6) for v in pose_flat]

            slam_data = {
                "tracking_quality": round(float(slam_result.tracking_quality), 3),
                "num_gaussians": int(slam_result.num_points),
                "camera_pose": pose_flat,
                "ply_version": self._ply_version,
            }

        try:
            await self._client.send_vision_status(
                state=self._state.value,
                detections=[],
                distance=us_distance,
                depth_strips=depth_strips,
                slam_data=slam_data,
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
            "tracking_lost", "num_keyframes",
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

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------
    async def _decision_loop(self) -> None:
        """Main autonomy loop: SLAM → depth → decide → act.

        Philosophy: always head toward the most free space.  The robot
        treats the SLAM-rendered depth map as a "clearance field" and steers
        toward whichever direction has the most room.  Speed is proportional
        to the distance ahead — lots of room = cruise, tight = creep.
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
            raw_center_dist = analysis["center_dist"]
            slam_result = analysis["slam_result"]
            self._frame_count += 1

            raw_us = self._latest_telemetry.get("distance", -1)
            us_distance = self._filter_distance(raw_us)
            if us_distance == float("inf"):
                us_distance = -1.0

            # SLAM depth strips are relative (no calibration needed for steering)
            strip_l, strip_c, strip_r = raw_depth_strips
            center_dist = raw_center_dist
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

            # === Initial scan: rotate in place to build panoramic map ===
            if not self._initial_scan_done and not self._debug:
                if self._scan_steps_remaining == 0:
                    self._scan_steps_remaining = INITIAL_SCAN_STEPS
                    log.info("Starting initial scan (%d steps)", INITIAL_SCAN_STEPS)

                # If tracking lost during scan, wait for recovery
                if slam_result is not None and slam_result.tracking_lost:
                    await self._stop()
                    await asyncio.sleep(0.3)
                    continue

                self._set_state(State.SCANNING)
                # High power, short pulse — overcome friction without overshooting
                await self._motor(-TURN_SPEED, TURN_SPEED)
                await asyncio.sleep(0.35)
                await self._stop()
                await asyncio.sleep(0.6)  # long pause for SLAM to track the new view
                self._scan_steps_remaining -= 1

                if self._scan_steps_remaining <= 0:
                    self._initial_scan_done = True
                    log.info("Initial scan complete — starting exploration")
                continue

            # During warmup (no depth yet), stay still
            if slam_result is None:
                self._set_state(State.SCANNING)
                if not self._debug:
                    await self._stop()
                await asyncio.sleep(0.1)
                continue

            # Tracking lost — stop and turn to find known features. NEVER reverse.
            if slam_result.tracking_lost:
                if not hasattr(self, '_lost_count'):
                    self._lost_count = 0
                self._lost_count += 1

                if not self._debug:
                    if self._lost_count == 1:
                        log.info("Tracking lost — stopping")
                        await self._stop()
                    elif self._lost_count % 15 == 0:
                        # Periodically turn in place to find known features
                        log.info("Tracking lost (%d frames) — turning to re-acquire", self._lost_count)
                        self._set_state(State.SCANNING)
                        await self._motor(-TURN_SPEED, TURN_SPEED)
                        await asyncio.sleep(0.35)
                        await self._stop()

                await asyncio.sleep(0.1)
                continue
            else:
                if getattr(self, '_lost_count', 0) > 0:
                    log.info("Tracking recovered after %d lost frames", self._lost_count)
                self._lost_count = 0

            # SLAM strips are relative — use for steering direction only
            # Ultrasonic is absolute cm — use for speed and stop/go decisions
            total_strip = strip_l + strip_c + strip_r

            if self._debug:
                if self._frame_count % 30 == 0:
                    log.info("US=%.1fcm | SLAM L/C/R=%.1f/%.1f/%.1f | points=%d",
                             us_distance, strip_l, strip_c, strip_r, slam_result.num_points)
                await asyncio.sleep(0.1)
                continue

            # === Stuck detection ===
            if slam_result is not None and not slam_result.tracking_lost:
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
                        await self._motor(-TURN_SPEED, TURN_SPEED)
                    else:
                        await self._motor(TURN_SPEED, -TURN_SPEED)
                    await asyncio.sleep(0.35)
                    await self._stop()
                    self._stuck_count = 0
                    self._last_nav_pose = None
                    await asyncio.sleep(0.4)
                    continue

            # === TIER 1: Emergency — ultrasonic says too close ===
            if 0 < us_distance < US_STOP:
                self._set_state(State.SCANNING)
                await self._stop()
                if strip_l > strip_r:
                    await self._motor(-TURN_SPEED, TURN_SPEED)
                else:
                    await self._motor(TURN_SPEED, -TURN_SPEED)
                await asyncio.sleep(0.35)
                await self._stop()
                await asyncio.sleep(0.4)
                continue

            # === TIER 2: Ultrasonic says getting close — slow zone ===
            # Clearance: 0 = at US_STOP, 1 = at US_CLEAR or beyond
            if us_distance > 0:
                clearance = max(0.0, min(1.0, (us_distance - US_STOP) / (US_CLEAR - US_STOP)))
            else:
                # No US reading — trust SLAM strips relatively: if all similar, assume clear
                clearance = 0.7  # moderate confidence

            # === Steering from SLAM depth strips (relative, no scale needed) ===
            if total_strip > 0.01:
                bias = (strip_r - strip_l) / total_strip
            else:
                bias = 0.0

            # Steer harder when closer to obstacles
            steer_strength = 0.3 + 0.5 * (1.0 - clearance)
            steer = bias * steer_strength

            # === Familiarity-based speed ===
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
            elif clearance < 0.7:
                self._set_state(State.CRUISING)
            else:
                self._set_state(State.CRUISING)

            # Pulse-and-coast
            pulse = PULSE_DURATION + 0.15 * familiarity
            await self._motor(left_speed, right_speed)
            await asyncio.sleep(pulse)
            await self._stop()
            await asyncio.sleep(COAST_PAUSE)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._running = True
        self._load_models()

        if self._record:
            self._init_recording()

        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(self._on_telemetry)

        await self._client.connect()
        if self._debug:
            log.info("Vision engine running — DEBUG MODE (no motor commands)")
        else:
            log.info("Vision engine running — 3DGS SLAM autonomous mode active")

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

    parser = argparse.ArgumentParser(description="Tank robot autonomous vision (3DGS SLAM)")
    parser.add_argument("--robot", default="", help="Robot WebSocket URL (or set ROBOT_URL env var)")
    parser.add_argument("--splat-dir", default="", help="Directory for PLY export (default: Phoenix static assets)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: run SLAM but send no motor commands")
    parser.add_argument("--record", action="store_true",
                        help="Record driving data (use with --debug while driving manually)")
    args = parser.parse_args()

    engine = VisionEngine(robot_url=args.robot, debug=args.debug, splat_dir=args.splat_dir, record=args.record)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
