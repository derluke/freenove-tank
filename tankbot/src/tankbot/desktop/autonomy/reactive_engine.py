"""Live reactive autonomy engine -- Phase 1 replacement for VisionEngine.

Tight loop, no SLAM, no FrontierPlanner, no scan-match in the hot path:

    frame (JPEG) -> decode -> depth inference -> ReactiveSafety.tick -> motor

The entire decision path is serialized: one depth inference per frame,
one reactive tick per depth, one motor command per tick. At ~23 Hz
(Depth Anything V2 Base on RTX 4070 Ti SUPER) this gives well under
50 ms end-to-end latency per control cycle.

Launch:
    tankbot-reactive --robot ws://<pi-ip>:9000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import time
from collections import deque

import cv2
import numpy as np

from tankbot.desktop.autonomy.depth import build_estimator
from tankbot.desktop.autonomy.reactive import (
    GyroPoseSource,
    ReactiveSafety,
    SafetyTick,
    load_extrinsics,
    load_intrinsics,
)
from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator
from tankbot.desktop.autonomy.reactive.grid import CELL_OCCUPIED
from tankbot.desktop.autonomy.reactive.policy import PolicyDecision, WanderState
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
)
from tankbot.desktop.autonomy.robot_client import RobotClient

log = logging.getLogger(__name__)

# Ultrasonic is reported in cm by the robot telemetry.
_US_MAX_TRUSTED_CM = 200.0
_DEGRADED_FPS_FLOOR = 10.0
_DEGRADED_FPS_RECOVER = 12.0
_LOOP_FPS_WINDOW = 20
_DEGRADED_CRUISE_DUTY = 1000
_NORMAL_GRID_PERSISTENCE_FRAMES = 7
_DEGRADED_GRID_PERSISTENCE_FRAMES = 1


class ReactiveEngine:
    """Minimal live autonomy: depth-based reactive wander."""

    def __init__(
        self,
        *,
        robot_url: str = "",
        debug: bool = False,
        depth_backend: str = "depth_anything_v2",
        intrinsics_path: str = "calibration/camera_intrinsics.yaml",
        extrinsics_path: str = "calibration/camera_extrinsics.yaml",
        depth_scale: float = 1.0,
    ) -> None:
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._client = RobotClient(robot_url)
        self._debug = debug
        self._depth_backend_name = depth_backend
        self._intrinsics_path = intrinsics_path
        self._extrinsics_path = extrinsics_path
        self._depth_scale = depth_scale

        self._latest_frame: bytes | None = None
        self._latest_telemetry: dict = {}
        # Motor state reported by the robot. Kept separate so vision-status
        # echoes (which the robot re-broadcasts to all clients) can't wipe
        # it — they have type="vision" and no motor field.
        self._latest_motor: dict = {"left": 0, "right": 0}
        self._running = False
        self._frame_count = 0
        self._degraded_mode = False
        self._tick_timestamps: deque[float] = deque(maxlen=_LOOP_FPS_WINDOW)

        # LED state to avoid spamming.
        self._last_led_payload: tuple | None = None

        # Gyro integration — only marked "active" once we've received samples.
        self._gyro = GyroIntegrator()
        self._imu_active = False

        self._last_left_duty = 0
        self._last_right_duty = 0

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    async def _on_frame(self, jpeg: bytes) -> None:
        self._latest_frame = jpeg

    async def _on_telemetry(self, data: dict) -> None:
        self._latest_telemetry = data
        motor = data.get("motor")
        if isinstance(motor, dict):
            self._latest_motor = motor

    async def _on_imu(self, data: dict) -> None:
        """High-rate IMU callback — accumulate gyro-Z into a yaw estimate.

        The robot piggybacks current motor state on each IMU message, so
        DR can track the actual motor duty at 50 Hz (rather than the 1 Hz
        telemetry loop, which misses most commands between samples).
        """
        gz = data.get("gz")
        if gz is None:
            return
        self._gyro.add_sample(float(gz))
        # Motor state: only update when present (stays correct on older robots).
        if "motor_left" in data and "motor_right" in data:
            self._latest_motor = {
                "left": int(data["motor_left"]),
                "right": int(data["motor_right"]),
            }
        if not self._imu_active:
            self._imu_active = True
            log.info("IMU stream active — gyro pose source online")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        log.info("Loading depth backend: %s", self._depth_backend_name)
        self._estimator = build_estimator(self._depth_backend_name)
        self._estimator.load()
        log.info("Depth backend ready.")

        self._intrinsics = load_intrinsics(self._intrinsics_path)
        self._extrinsics = load_extrinsics(self._extrinsics_path)

        self._proj_cfg = ProjectionConfig(depth_scale=self._depth_scale)
        if self._depth_scale != 1.0:
            log.info("Depth scale: %.3f", self._depth_scale)

        self._pose_source = GyroPoseSource(self._gyro)

        self._safety = ReactiveSafety(
            intrinsics=self._intrinsics,
            extrinsics=self._extrinsics,
            pose_source=self._pose_source,
            projection_config=self._proj_cfg,
        )
        self._safety.grid.set_persistence_frames(_NORMAL_GRID_PERSISTENCE_FRAMES)
        log.info("Gyro pose source + reactive safety initialized.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_frame(self, jpeg: bytes) -> np.ndarray | None:
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return bgr

    def _get_ultrasonic_m(self) -> float | None:
        raw_cm = self._latest_telemetry.get("distance", -1)
        if not isinstance(raw_cm, (int, float)) or raw_cm <= 0 or raw_cm > _US_MAX_TRUSTED_CM:
            return None
        return float(raw_cm) / 100.0

    async def _send_motor(self, left: int, right: int) -> None:
        if self._debug:
            return
        await self._client.send_motor(left, right)

    def _update_loop_mode(self, t_now: float) -> None:
        """Enter/exit degraded mode based on sustained control-loop FPS."""
        self._tick_timestamps.append(t_now)
        if len(self._tick_timestamps) < 2:
            return
        dt = self._tick_timestamps[-1] - self._tick_timestamps[0]
        if dt <= 1e-6:
            return
        fps = (len(self._tick_timestamps) - 1) / dt

        if not self._degraded_mode and len(self._tick_timestamps) >= 10 and fps < _DEGRADED_FPS_FLOOR:
            self._degraded_mode = True
            self._safety.grid.set_persistence_frames(_DEGRADED_GRID_PERSISTENCE_FRAMES)
            log.warning(
                "Reactive loop degraded: %.1f FPS < %.1f. Grid persistence -> %d frame, forward duty capped.",
                fps,
                _DEGRADED_FPS_FLOOR,
                _DEGRADED_GRID_PERSISTENCE_FRAMES,
            )
        elif self._degraded_mode and fps > _DEGRADED_FPS_RECOVER:
            self._degraded_mode = False
            self._safety.grid.set_persistence_frames(_NORMAL_GRID_PERSISTENCE_FRAMES)
            log.info(
                "Reactive loop recovered: %.1f FPS > %.1f. Grid persistence -> %d frames.",
                fps,
                _DEGRADED_FPS_RECOVER,
                _NORMAL_GRID_PERSISTENCE_FRAMES,
            )

    def _apply_degraded_cap(self, decision: PolicyDecision) -> PolicyDecision:
        if not self._degraded_mode or decision.state != WanderState.FORWARD:
            return decision
        left = max(-_DEGRADED_CRUISE_DUTY, min(_DEGRADED_CRUISE_DUTY, decision.left_duty))
        right = max(-_DEGRADED_CRUISE_DUTY, min(_DEGRADED_CRUISE_DUTY, decision.right_duty))
        if left == decision.left_duty and right == decision.right_duty:
            return decision
        return PolicyDecision(
            left_duty=left,
            right_duty=right,
            reason=f"{decision.reason} [degraded fps cap]",
            state=decision.state,
        )

    async def _update_leds(self, decision: PolicyDecision, blocked: bool) -> None:
        """Expressive LED feedback using all 4 LEDs individually.

        LED layout (looking at the robot from above):
            0, 1 = left side    2, 3 = right side

        Effects:
        - Forward: green pulse (brightness breathes with frame count)
        - Turning left:  left LEDs amber blink, right LEDs dim
        - Turning right: right LEDs amber blink, left LEDs dim
        - Reversing: all LEDs red, alternating pairs blink
        - Blocked: all LEDs fast red strobe
        - Stopped: dim white
        """
        if self._debug:
            return

        fc = self._frame_count
        state = decision.state

        if state == WanderState.REVERSING:
            # Alternating red pairs: 0,1 on even ticks, 2,3 on odd ticks.
            if (fc // 3) % 2 == 0:
                payload = ("split", 255, 40, 0, 0b0011, 60, 10, 0, 0b1100)
            else:
                payload = ("split", 60, 10, 0, 0b0011, 255, 40, 0, 0b1100)
        elif blocked:
            # Fast red strobe: all on/off every 2 frames.
            payload = ("all", 255, 20, 10, 0xF) if (fc // 2) % 2 == 0 else ("all", 40, 5, 2, 0xF)
        elif state == WanderState.TURNING:
            # Directional indicator: amber blink on the turn side.
            blink_on = (fc // 4) % 2 == 0  # ~3 Hz blink at 23 fps
            if decision.left_duty < 0:
                # Turning left: left LEDs amber blink, right dim.
                if blink_on:
                    payload = ("split", 255, 140, 0, 0b0011, 40, 30, 0, 0b1100)
                else:
                    payload = ("split", 60, 30, 0, 0b0011, 40, 30, 0, 0b1100)
            else:
                # Turning right: right LEDs amber blink, left dim.
                if blink_on:
                    payload = ("split", 40, 30, 0, 0b0011, 255, 140, 0, 0b1100)
                else:
                    payload = ("split", 40, 30, 0, 0b0011, 60, 30, 0, 0b1100)
        elif state == WanderState.FORWARD:
            # Green breathing: brightness oscillates with a sine wave.
            breath = 0.5 + 0.5 * math.sin(fc * 0.15)
            g = int(80 + 140 * breath)
            payload = ("all", 0, g, 20, 0xF)
        else:
            payload = ("all", 50, 50, 50, 0xF)

        if payload != self._last_led_payload:
            self._last_led_payload = payload
            if payload[0] == "split":
                # Two LED groups with different colors.
                _, r1, g1, b1, m1, r2, g2, b2, m2 = payload
                await self._client.send_led(r1, g1, b1, m1)
                await self._client.send_led(r2, g2, b2, m2)
            else:
                _, r, g, b, m = payload
                await self._client.send_led(r, g, b, m)

    def _encode_depth_image(self, depth_m: np.ndarray) -> str:
        """Colorize + JPEG-encode + base64 a depth map for the dashboard."""
        clipped = np.clip(depth_m, 0.0, 5.0)
        norm = (clipped / 5.0 * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        small = cv2.resize(color, (320, 240), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return base64.b64encode(buf).decode("ascii")

    def _encode_grid_image(self) -> str:
        """Render the reactive occupancy grid for the dashboard."""
        cells = self._safety.grid.cells
        size = self._safety.grid.size
        bgr = np.full((size, size, 3), 40, dtype=np.uint8)
        bgr[cells == CELL_OCCUPIED] = (0, 0, 220)
        bgr = np.ascontiguousarray(bgr[::-1, :, :])
        center = size // 2
        cv2.circle(bgr, (center, center), 2, (0, 220, 0), -1)
        bgr = cv2.resize(bgr, (480, 480), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("ascii")

    async def _send_dashboard_status(
        self,
        tick: SafetyTick,
        decision: PolicyDecision,
        depth_ms: float,
        us_m: float | None,
        depth_m: np.ndarray,
    ) -> None:
        """Push reactive state to the web dashboard via the robot WS."""
        state_str = decision.state.value  # "forward" / "turning" / "stopped"
        depth_b64 = self._encode_depth_image(depth_m)

        # Encode grid every 5th frame to save bandwidth.
        grid_b64 = self._encode_grid_image() if self._frame_count % 5 == 0 else None

        pose = tick.pose
        xy = pose.xy_m
        pose_data = {
            "x": round(xy[0], 3) if xy is not None else None,
            "y": round(xy[1], 3) if xy is not None else None,
            "yaw_deg": round(math.degrees(pose.yaw_rad), 1),
            "health": pose.health.value,
            "imu_active": self._imu_active,
            "gyro_yaw_deg": round(math.degrees(self._gyro.yaw_rad), 1) if self._imu_active else None,
        }

        try:
            autonomy_data: dict = {
                "engine": "reactive",
                "phase": "reactive",
                "goal": "wander",
                "behavior": state_str,
                "status_detail": decision.reason,
                "depth_image": depth_b64,
                # Reactive-specific fields.
                "stop_zone_blocked": tick.stop_zone_blocked,
                "stop_zone_count": tick.stop_zone_count,
                "near_field_veto": tick.near_field_veto,
                "clearance_forward_m": round(tick.clearance_forward_m, 2),
                "clearance_left_m": round(tick.clearance_left_m, 2),
                "clearance_right_m": round(tick.clearance_right_m, 2),
                "obstacle_points": tick.n_obstacle_points,
                "motor_left": decision.left_duty,
                "motor_right": decision.right_duty,
                "depth_ms": round(depth_ms, 1),
                "projection_ms": round(tick.projection_ms, 1),
                "grid_ms": round(tick.grid_ms, 1),
                "total_ms": round(tick.total_ms, 1),
                "degraded_mode": self._degraded_mode,
                "ultrasonic_m": round(us_m, 2) if us_m is not None else None,
                "frame_count": self._frame_count,
                # Phase 1: gyro pose + local safety grid.
                "pose": pose_data,
                "grid_cells": int((self._safety.grid.cells == CELL_OCCUPIED).sum()),
                "grid_persistence_frames": self._safety.grid.persistence_frames,
            }
            if grid_b64 is not None:
                autonomy_data["map_image"] = grid_b64
                autonomy_data["map_mode"] = "robot"

            await self._client.send_vision_status(
                state=state_str,
                detections=[],
                distance=self._latest_telemetry.get("distance", -1),
                autonomy_data=autonomy_data,
            )
        except Exception:
            if self._frame_count % 100 == 0:
                log.exception("Failed to send dashboard status")

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------
    async def _decision_loop(self) -> None:
        loop = asyncio.get_running_loop()

        while self._running:
            frame = self._latest_frame
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            self._latest_frame = None
            self._frame_count += 1

            # Decode JPEG.
            bgr = self._decode_frame(frame)
            if bgr is None:
                continue

            # Depth inference (GPU-bound, ~43 ms).
            t0 = time.monotonic()
            depth_result = await loop.run_in_executor(None, self._estimator.infer, bgr)
            depth_ms = (time.monotonic() - t0) * 1000.0

            depth_m = depth_result.depth_m

            # Ultrasonic.
            us_m = self._get_ultrasonic_m()

            # Reactive tick.
            tick = self._safety.tick(depth_m, ultrasonic_m=us_m)
            self._update_loop_mode(time.monotonic())

            # Motor command.
            decision = self._apply_degraded_cap(tick.decision)
            await self._send_motor(decision.left_duty, decision.right_duty)

            # DR source: in live mode the engine's own decision is ground
            # truth (and low-latency). In debug mode the engine doesn't
            # send motor commands — the dashboard does — so DR has to read
            # the robot's actual motor state from telemetry.
            if self._debug:
                self._last_left_duty = int(self._latest_motor.get("left", 0))
                self._last_right_duty = int(self._latest_motor.get("right", 0))
            else:
                self._last_left_duty = decision.left_duty
                self._last_right_duty = decision.right_duty
            # LED feedback.
            await self._update_leds(decision, tick.stop_zone_blocked)

            # Dashboard status (every frame so the web UI stays live).
            await self._send_dashboard_status(tick, decision, depth_ms, us_m, depth_m)

            # Periodic log.
            if self._frame_count % 30 == 0:
                # Log the center-strip depth median alongside the ultrasonic
                # reading so the user can calibrate --depth-scale.
                h, w = depth_m.shape
                center_strip = depth_m[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
                depth_median = float(np.median(center_strip))
                imu_tag = "imu" if self._imu_active else "no-imu"
                pose = tick.pose
                if pose.xy_m is not None:
                    pose_str = f"pose=({pose.xy_m[0]:+.2f},{pose.xy_m[1]:+.2f})m/{math.degrees(pose.yaw_rad):+.0f}°"
                else:
                    pose_str = f"pose=None/{math.degrees(pose.yaw_rad):+.0f}°"
                log.info(
                    "[%d] depth=%.0fms react=%.1fms %s L=%+d R=%+d fwd=%.2fm us=%s pts=%d dmed=%.2fm %s health=%s %s",
                    self._frame_count,
                    depth_ms,
                    tick.total_ms,
                    "BLOCKED" if tick.stop_zone_blocked else "clear",
                    decision.left_duty,
                    decision.right_duty,
                    tick.clearance_forward_m,
                    f"{us_m:.2f}m" if us_m is not None else "n/a",
                    tick.n_obstacle_points,
                    depth_median,
                    pose_str,
                    pose.health.value,
                    f"{imu_tag}{' degraded' if self._degraded_mode else ''}",
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._running = True
        self._load_models()

        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(self._on_telemetry)
        self._client.on_imu(self._on_imu)

        await self._client.connect()
        if self._debug:
            log.info("Reactive engine running -- DEBUG MODE (no motor commands)")
        else:
            log.info("Reactive engine running -- motors LIVE")

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
            self._estimator.unload()
            await self._client.close()
            log.info("Reactive engine stopped")


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Reactive autonomy (Phase 1 -- no SLAM)")
    parser.add_argument("--robot", default="", help="Robot WebSocket URL (or set ROBOT_URL env var)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: no motor commands")
    parser.add_argument("--backend", default="depth_anything_v2", help="Depth backend name")
    parser.add_argument("--intrinsics", default="calibration/camera_intrinsics.yaml")
    parser.add_argument("--extrinsics", default="calibration/camera_extrinsics.yaml")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth scale correction. Calibrate: face a hard flat wall, "
        "set depth_scale = ultrasonic_m / depth_model_median. "
        "Values < 1 mean the model over-estimates distance.",
    )
    args = parser.parse_args()

    engine = ReactiveEngine(
        robot_url=args.robot,
        debug=args.debug,
        depth_backend=args.backend,
        intrinsics_path=args.intrinsics,
        extrinsics_path=args.extrinsics,
        depth_scale=args.depth_scale,
    )
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
