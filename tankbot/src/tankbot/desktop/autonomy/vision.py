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

# Speeds (duty values, max 4095)
CRUISE_SPEED = 1200
SLOW_SPEED = 1100
TURN_SPEED = 1300
BACKUP_SPEED = 1200
MIN_DUTY = 1000

# Depth thresholds in METERS (real distances from SLAM-rendered depth)
DEPTH_STOP_M = 0.40       # closer than 40 cm → back up
DEPTH_SLOW_M = 1.00       # closer than 1 m → slow down and steer
DEPTH_CLEAR_M = 1.50      # farther than 1.5 m → considered clear

# We divide the frame into three vertical strips for steering decisions
STRIP_LEFT = (0, FRAME_WIDTH // 3)
STRIP_CENTER = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)
STRIP_RIGHT = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)

# Ultrasonic thresholds (cm)
US_STOP = 20
US_SLOW = 50
US_MAX_TRUSTED = 200

# Timing (seconds)
BACKUP_DURATION = 0.4
TURN_AFTER_BACKUP = 0.35
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


# Force debug mode until SLAM tracking is validated
FORCE_DEBUG = True


class VisionEngine:
    def __init__(self, robot_url: str = "", debug: bool = False,
                 splat_dir: str = "") -> None:
        debug = debug or FORCE_DEBUG
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

            depth_strips = analysis["depth_strips"]
            center_dist = analysis["center_dist"]
            slam_result = analysis["slam_result"]
            strip_l, strip_c, strip_r = depth_strips
            self._frame_count += 1

            raw_us = self._latest_telemetry.get("distance", -1)
            us_distance = self._filter_distance(raw_us)
            # Clamp to finite value — JSON can't represent Infinity
            if us_distance == float("inf"):
                us_distance = -1.0

            # Export PLY periodically (in executor since it touches disk)
            if slam_result is not None:
                await loop.run_in_executor(None, self._maybe_export_ply)

            await self._send_vision_status(us_distance, depth_strips, slam_result)

            # During warmup (no depth yet), stay still
            if slam_result is None:
                self._set_state(State.SCANNING)
                if not self._debug:
                    await self._stop()
                await asyncio.sleep(0.1)
                continue

            # After bootstrap period, gate on tracking quality
            bootstrapping = self._frame_count < TRACKING_BOOTSTRAP_FRAMES
            if not bootstrapping and slam_result.tracking_quality < TRACKING_LOSS_THRESHOLD:
                self._set_state(State.STOPPED)
                if self._send_counter % 30 == 0:
                    log.warning(
                        "Tracking quality low (%.2f), gaussians=%d, stopping",
                        slam_result.tracking_quality, slam_result.num_points,
                    )
                if not self._debug:
                    await self._stop()
                await asyncio.sleep(0.1)
                continue

            # The most free space in any direction
            best_strip = max(strip_l, strip_c, strip_r)
            # The least free space (tightest constraint)
            worst_strip = min(strip_l, strip_c, strip_r)

            if self._debug:
                if self._frame_count % 30 == 0:
                    log.info(
                        "depth L=%.2fm C=%.2fm R=%.2fm | center=%.2fm | US=%.1fcm | points=%d | quality=%.2f",
                        strip_l, strip_c, strip_r, center_dist,
                        us_distance if us_distance != float("inf") else -1,
                        slam_result.num_points,
                        slam_result.tracking_quality,
                    )
                await asyncio.sleep(0.1)
                continue

            # === TIER 1: Emergency — something right in our face ===
            emergency_us = 0 < us_distance < US_STOP
            emergency_depth = worst_strip < DEPTH_STOP_M

            if emergency_us or emergency_depth:
                self._set_state(State.BACKING_UP)
                await self._stop()
                await asyncio.sleep(0.05)
                await self._motor(-BACKUP_SPEED, -BACKUP_SPEED)
                await asyncio.sleep(BACKUP_DURATION)
                # Turn toward the most open side
                if strip_l > strip_r:
                    await self._motor(-TURN_SPEED, TURN_SPEED)
                else:
                    await self._motor(TURN_SPEED, -TURN_SPEED)
                await asyncio.sleep(TURN_AFTER_BACKUP)
                self._last_turn_dir *= -1
                await self._stop()
                self._set_state(State.SCANNING)
                await asyncio.sleep(SCAN_PAUSE)
                continue

            # === TIER 2: Boxed in — no direction has much room ===
            if best_strip < DEPTH_SLOW_M:
                # Turn in place toward whichever strip is most open
                self._set_state(State.SCANNING)
                if strip_l >= strip_r:
                    await self._motor(-TURN_SPEED, TURN_SPEED)
                else:
                    await self._motor(TURN_SPEED, -TURN_SPEED)
                await asyncio.sleep(0.15)
                continue

            # === TIER 3: Navigate toward the most free space ===
            clearance = max(0.0, min(1.0,
                (center_dist - DEPTH_STOP_M) / (DEPTH_CLEAR_M - DEPTH_STOP_M)
            ))
            base_speed = int(MIN_DUTY + (CRUISE_SPEED - MIN_DUTY) * clearance)

            total = strip_l + strip_c + strip_r
            if total > 0.01:
                bias = (strip_r - strip_l) / total
            else:
                bias = 0.0

            steer = bias * 0.5
            left_speed = int(base_speed * (1 + steer))
            right_speed = int(base_speed * (1 - steer))

            left_speed = max(MIN_DUTY, min(left_speed, CRUISE_SPEED))
            right_speed = max(MIN_DUTY, min(right_speed, CRUISE_SPEED))

            if center_dist < DEPTH_SLOW_M:
                self._set_state(State.AVOIDING)
            else:
                self._set_state(State.CRUISING)

            await self._motor(left_speed, right_speed)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._running = True
        self._load_models()

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
    args = parser.parse_args()

    engine = VisionEngine(robot_url=args.robot, debug=args.debug, splat_dir=args.splat_dir)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
