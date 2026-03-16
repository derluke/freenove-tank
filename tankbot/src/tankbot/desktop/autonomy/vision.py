"""Vision pipeline — runs YOLO26 + Depth Pro on robot camera frames using the desktop GPU.

Connects to the robot via WebSocket, receives frames, runs object detection
and metric depth estimation, and sends motor commands back to navigate
autonomously.

Behaviour: cautious exploration.  The robot creeps forward slowly, scans for
obstacles with both camera (YOLO + metric depth) and ultrasonic, and steers /
stops / backs up as needed.

Sensor fusion:
  - YOLO26: identifies objects + bounding boxes
  - Depth Pro (Apple): metric depth map in *meters* from a single frame —
    detects every surface including walls, blankets, soft obstacles
  - Ultrasonic: forward-only proximity, unreliable on soft surfaces
  - Depth map is the primary obstacle sensor; YOLO + ultrasonic augment it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image

from tankbot.desktop.autonomy.robot_client import RobotClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
FRAME_WIDTH = 400
FRAME_HEIGHT = 300

# Speeds (duty values, max 4095)
CRUISE_SPEED = 1200
SLOW_SPEED = 1100
TURN_SPEED = 1300
BACKUP_SPEED = 1200
MIN_DUTY = 1000

# Depth thresholds in METERS (real distances from Depth Pro)
DEPTH_STOP_M = 0.40       # closer than 40 cm → back up
DEPTH_SLOW_M = 1.00       # closer than 1 m → slow down and steer
DEPTH_CLEAR_M = 1.50      # farther than 1.5 m → considered clear

# We divide the frame into three vertical strips for steering decisions
STRIP_LEFT = (0, FRAME_WIDTH // 3)
STRIP_CENTER = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)
STRIP_RIGHT = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)

# YOLO thresholds (secondary to depth — used for labelling + dashboard)
MIN_CONFIDENCE = 0.35
LARGE_OBJECT_AREA = 0.04

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

# ---------------------------------------------------------------------------
# Exploration map
# ---------------------------------------------------------------------------
MAP_CELL_CM = 10
MAP_SIZE = 100
CM_PER_SECOND_AT_CRUISE = 15.0
TRACK_WIDTH_CM = 16.0


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class State(str, Enum):
    SCANNING = "scanning"
    CRUISING = "cruising"
    AVOIDING = "avoiding"
    BACKING_UP = "backing_up"
    STOPPED = "stopped"


class ExplorationMap:
    """Simple 2D occupancy grid built from dead-reckoning + detections."""

    def __init__(self, size: int = MAP_SIZE, cell_cm: float = MAP_CELL_CM) -> None:
        self.size = size
        self.cell_cm = cell_cm
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.x = size / 2 * cell_cm
        self.y = size / 2 * cell_cm
        self.heading = math.pi / 2
        self._last_update = time.monotonic()
        self._mark(self.x, self.y, 1)

    def update_pose(self, left_duty: int, right_duty: int) -> None:
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        if dt <= 0 or dt > 1.0:
            return
        max_duty = 4095
        left_speed = (left_duty / max_duty) * CM_PER_SECOND_AT_CRUISE * (max_duty / CRUISE_SPEED)
        right_speed = (right_duty / max_duty) * CM_PER_SECOND_AT_CRUISE * (max_duty / CRUISE_SPEED)
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / TRACK_WIDTH_CM
        self.heading += omega * dt
        self.x += v * dt * math.cos(self.heading)
        self.y += v * dt * math.sin(self.heading)
        self._mark(self.x, self.y, 1)

    def mark_obstacle_ahead(self, distance_cm: float) -> None:
        if distance_cm <= 0 or distance_cm > 300:
            return
        ox = self.x + distance_cm * math.cos(self.heading)
        oy = self.y + distance_cm * math.sin(self.heading)
        self._mark(ox, oy, -1)

    def mark_obstacle_at_angle(self, distance_cm: float, angle_offset: float) -> None:
        if distance_cm <= 0 or distance_cm > 300:
            return
        heading = self.heading + angle_offset
        ox = self.x + distance_cm * math.cos(heading)
        oy = self.y + distance_cm * math.sin(heading)
        self._mark(ox, oy, -1)

    def _mark(self, x_cm: float, y_cm: float, value: int) -> None:
        gx = int(x_cm / self.cell_cm)
        gy = int(y_cm / self.cell_cm)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            self.grid[gy, gx] = value

    def to_sparse(self) -> dict:
        visited = []
        obstacles = []
        for gy in range(self.size):
            for gx in range(self.size):
                v = self.grid[gy, gx]
                if v == 1:
                    visited.append([gx, gy])
                elif v == -1:
                    obstacles.append([gx, gy])
        return {
            "size": self.size,
            "cell_cm": self.cell_cm,
            "robot": [
                round(self.x / self.cell_cm, 1),
                round(self.y / self.cell_cm, 1),
                round(math.degrees(self.heading), 1),
            ],
            "visited": visited,
            "obstacles": obstacles,
        }


class DepthEstimator:
    """Apple Depth Pro — metric monocular depth estimation.

    Outputs a depth map in meters. No relative nonsense.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self._model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(self._device).eval()
        log.info("Depth Pro loaded on %s (fp16 + SDPA)", self._device)

    def estimate(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return depth map in meters (H×W float32, lower = closer)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = self._processor(images=pil_img, return_tensors="pt").to(
            device=self._device, dtype=torch.float16
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        post = self._processor.post_process_depth_estimation(
            outputs, target_sizes=[(img_bgr.shape[0], img_bgr.shape[1])]
        )
        depth_meters = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
        return depth_meters

    def get_strip_distances(self, depth_map: np.ndarray) -> tuple[float, float, float]:
        """Get the nearest obstacle distance (meters) in left, center, right strips.

        Uses the middle band of the frame (rows 30%-80%) to avoid
        floor right in front of the camera and ceiling.
        Returns the 10th percentile (closest stuff) per strip.
        """
        h = depth_map.shape[0]
        roi = depth_map[int(h * 0.3):int(h * 0.8), :]

        left = float(np.percentile(roi[:, STRIP_LEFT[0]:STRIP_LEFT[1]], 10))
        center = float(np.percentile(roi[:, STRIP_CENTER[0]:STRIP_CENTER[1]], 10))
        right = float(np.percentile(roi[:, STRIP_RIGHT[0]:STRIP_RIGHT[1]], 10))
        return round(left, 2), round(center, 2), round(right, 2)

    def get_center_distance(self, depth_map: np.ndarray) -> float:
        """Nearest obstacle distance (meters) in the central region."""
        h, w = depth_map.shape
        roi = depth_map[int(h * 0.3):int(h * 0.8), int(w * 0.3):int(w * 0.7)]
        return round(float(np.percentile(roi, 10)), 2)


class VisionEngine:
    def __init__(self, robot_url: str = "", model: str = "yolo26l.pt",
                 debug: bool = False) -> None:
        import os
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._client = RobotClient(robot_url)
        self._model_path = model
        self._yolo = None
        self._depth = DepthEstimator()
        self._latest_frame: bytes | None = None
        self._latest_telemetry: dict = {}
        self._running = False
        self._debug = debug
        self._state = State.SCANNING
        self._state_since = time.monotonic()
        self._last_turn_dir: int = 1
        self._current_motor = (0, 0)
        self._map = ExplorationMap()
        self._map_send_counter = 0
        self._us_history: list[float] = []

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        from ultralytics import YOLO
        self._yolo = YOLO(self._model_path)
        log.info("YOLO model loaded: %s", self._model_path)
        self._depth.load()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    async def _on_frame(self, jpeg: bytes) -> None:
        self._latest_frame = jpeg

    async def _on_telemetry(self, data: dict) -> None:
        self._latest_telemetry = data

    # ------------------------------------------------------------------
    # Detection + Depth
    # ------------------------------------------------------------------
    def _analyse_frame(self, jpeg: bytes) -> dict:
        """Run YOLO + Depth Pro on a frame."""
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"detections": [], "depth_strips": (99, 99, 99), "center_dist": 99}

        # YOLO detection
        results = self._yolo(img, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                if conf < MIN_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / (img.shape[0] * img.shape[1])
                if area_ratio >= LARGE_OBJECT_AREA:
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 2),
                        "bbox": (round(x1), round(y1), round(x2), round(y2)),
                        "center_x": (x1 + x2) / 2,
                        "area_ratio": round(area_ratio, 3),
                    })

        # Depth Pro — metric depth in meters
        depth_map = self._depth.estimate(img)

        # Enrich detections with real distance in meters
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            roi = depth_map[y1:y2, x1:x2]
            if roi.size > 0:
                det["distance_m"] = round(float(np.percentile(roi, 10)), 2)
            else:
                det["distance_m"] = 99.0

        # Strip distances for steering
        strip_l, strip_c, strip_r = self._depth.get_strip_distances(depth_map)
        center_dist = self._depth.get_center_distance(depth_map)

        return {
            "detections": detections,
            "depth_strips": (strip_l, strip_c, strip_r),
            "center_dist": center_dist,
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

    def _update_map(self, detections: list[dict], us_distance: float,
                    depth_strips: tuple[float, float, float]) -> None:
        self._map.update_pose(*self._current_motor)

        if 0 < us_distance < 200:
            self._map.mark_obstacle_ahead(us_distance)

        for det in detections:
            cx = det["center_x"]
            angle_offset = ((cx - FRAME_WIDTH / 2) / FRAME_WIDTH) * 1.05
            dist_cm = det.get("distance_m", 99) * 100
            self._map.mark_obstacle_at_angle(dist_cm, -angle_offset)

        # Mark obstacles from depth strips
        for dist_m, angle in [
            (depth_strips[0], 0.4),
            (depth_strips[1], 0.0),
            (depth_strips[2], -0.4),
        ]:
            if dist_m < DEPTH_SLOW_M:
                self._map.mark_obstacle_at_angle(dist_m * 100, angle)

    async def _send_vision_status(self, detections: list[dict], us_distance: float,
                                  depth_strips: tuple[float, float, float]) -> None:
        self._map_send_counter += 1
        map_data = None
        if self._map_send_counter >= 5:
            self._map_send_counter = 0
            map_data = self._map.to_sparse()

        await self._client.send_vision_status(
            state=self._state.value,
            detections=[
                {
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "area": d["area_ratio"],
                    "depth": d.get("distance_m", 99),
                    "x1": round(d["bbox"][0] / FRAME_WIDTH * 100, 1),
                    "y1": round(d["bbox"][1] / FRAME_HEIGHT * 100, 1),
                    "x2": round(d["bbox"][2] / FRAME_WIDTH * 100, 1),
                    "y2": round(d["bbox"][3] / FRAME_HEIGHT * 100, 1),
                }
                for d in detections
            ],
            distance=us_distance,
            depth_strips=depth_strips,
            map_data=map_data,
        )

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------
    async def _decision_loop(self) -> None:
        """Main autonomy loop: detect → estimate depth → decide → act.

        Philosophy: always head toward the most free space.  The robot
        treats the depth map as a "clearance field" and steers toward
        whichever direction has the most room.  Speed is proportional
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

            detections = analysis["detections"]
            depth_strips = analysis["depth_strips"]
            center_dist = analysis["center_dist"]
            strip_l, strip_c, strip_r = depth_strips

            raw_us = self._latest_telemetry.get("distance", -1)
            us_distance = self._filter_distance(raw_us)

            self._update_map(
                detections,
                us_distance if us_distance != float("inf") else -1,
                depth_strips,
            )
            await self._send_vision_status(detections, us_distance, depth_strips)

            # The most free space in any direction
            best_strip = max(strip_l, strip_c, strip_r)
            # The least free space (tightest constraint)
            worst_strip = min(strip_l, strip_c, strip_r)

            if self._debug:
                log.info(
                    "depth L=%.2fm C=%.2fm R=%.2fm | center=%.2fm | US=%.1fcm | objects=%d",
                    strip_l, strip_c, strip_r, center_dist,
                    us_distance if us_distance != float("inf") else -1,
                    len(detections),
                )
                if worst_strip < DEPTH_STOP_M or (0 < us_distance < US_STOP):
                    self._set_state(State.BACKING_UP)
                elif best_strip < DEPTH_SLOW_M:
                    self._set_state(State.STOPPED)
                elif center_dist < DEPTH_SLOW_M:
                    self._set_state(State.AVOIDING)
                else:
                    self._set_state(State.CRUISING)
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
            #
            # Speed: proportional to how much room is ahead (center).
            # Steering: bias toward the strip with the most distance.
            #
            # This means the robot naturally flows into open areas
            # and gently curves away from walls — no hard "avoid" mode.

            # Speed: scale between MIN_DUTY and CRUISE_SPEED based on
            # center clearance.  At DEPTH_STOP_M → MIN_DUTY, at
            # DEPTH_CLEAR_M or more → CRUISE_SPEED.
            clearance = max(0.0, min(1.0,
                (center_dist - DEPTH_STOP_M) / (DEPTH_CLEAR_M - DEPTH_STOP_M)
            ))
            base_speed = int(MIN_DUTY + (CRUISE_SPEED - MIN_DUTY) * clearance)

            # Steering: compute a bias from -1 (turn left) to +1 (turn right)
            # based on where the most space is.
            total = strip_l + strip_c + strip_r
            if total > 0.01:
                # Weighted: right pulls positive, left pulls negative
                bias = (strip_r - strip_l) / total
            else:
                bias = 0.0

            # Convert bias to differential drive.
            # bias=0 → straight, bias=+1 → hard right, bias=-1 → hard left
            # We scale the steering strength by how different the sides are.
            steer = bias * 0.5  # 0..0.5 range, keeps it smooth
            left_speed = int(base_speed * (1 + steer))
            right_speed = int(base_speed * (1 - steer))

            # Clamp
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
            log.info("Vision engine running — cautious autonomous mode active")

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

    parser = argparse.ArgumentParser(description="Tank robot autonomous vision")
    parser.add_argument("--robot", default="", help="Robot WebSocket URL (or set ROBOT_URL env var)")
    parser.add_argument("--model", default="yolo26l.pt", help="YOLO model path")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: run detection + depth but send no motor commands")
    args = parser.parse_args()

    engine = VisionEngine(robot_url=args.robot, model=args.model, debug=args.debug)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
