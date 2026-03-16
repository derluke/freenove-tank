"""Vision pipeline — runs YOLO on robot camera frames using the desktop GPU.

Connects to the robot via WebSocket, receives frames, runs object detection,
and sends motor commands back to navigate autonomously.
"""

from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np

from tankbot.desktop.autonomy.robot_client import RobotClient

log = logging.getLogger(__name__)

# Detection parameters
OBSTACLE_CLASSES = {"person", "chair", "table", "couch", "bed", "dog", "cat"}
FRAME_WIDTH = 400
CENTER_TOLERANCE = 60  # pixels from center before steering
MIN_AREA_RATIO = 0.3  # object takes up 30% of frame → too close
CRUISE_SPEED = 1500
TURN_SPEED = 2000


class VisionEngine:
    def __init__(self, robot_url: str = "", model: str = "yolo11n.pt") -> None:
        import os
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._client = RobotClient(robot_url)
        self._model_path = model
        self._model = None
        self._latest_frame: bytes | None = None
        self._latest_telemetry: dict = {}
        self._running = False

    def _load_model(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._model_path)
        log.info("YOLO model loaded: %s", self._model_path)

    async def _on_frame(self, jpeg: bytes) -> None:
        self._latest_frame = jpeg

    async def _on_telemetry(self, data: dict) -> None:
        self._latest_telemetry = data

    def _detect(self, jpeg: bytes) -> list[dict]:
        """Run YOLO on a JPEG frame, return detections."""
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        results = self._model(img, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center_x": (x1 + x2) / 2,
                    "area_ratio": ((x2 - x1) * (y2 - y1)) / (img.shape[0] * img.shape[1]),
                })
        return detections

    async def _decision_loop(self) -> None:
        """Main autonomy loop: detect → decide → act."""
        while self._running:
            if self._latest_frame is None:
                await asyncio.sleep(0.05)
                continue

            frame = self._latest_frame
            self._latest_frame = None

            # Run detection in thread pool (GPU-bound but avoids blocking event loop)
            loop = asyncio.get_running_loop()
            detections = await loop.run_in_executor(None, self._detect, frame)

            # Find obstacles
            obstacles = [d for d in detections if d["class"] in OBSTACLE_CLASSES and d["confidence"] > 0.5]

            distance = self._latest_telemetry.get("distance", 999)

            # Decision logic
            if distance < 30 or any(d["area_ratio"] > MIN_AREA_RATIO for d in obstacles):
                # Too close — back up and turn
                await self._client.send_motor(-CRUISE_SPEED, -CRUISE_SPEED)
                await asyncio.sleep(0.3)
                await self._client.send_motor(-TURN_SPEED, TURN_SPEED)
                await asyncio.sleep(0.3)
            elif obstacles:
                # Steer away from the closest obstacle
                closest = max(obstacles, key=lambda d: d["area_ratio"])
                cx = closest["center_x"]
                frame_center = FRAME_WIDTH / 2

                if cx < frame_center - CENTER_TOLERANCE:
                    # Obstacle on left → steer right
                    await self._client.send_motor(CRUISE_SPEED, CRUISE_SPEED // 2)
                elif cx > frame_center + CENTER_TOLERANCE:
                    # Obstacle on right → steer left
                    await self._client.send_motor(CRUISE_SPEED // 2, CRUISE_SPEED)
                else:
                    # Obstacle dead ahead → slow approach
                    await self._client.send_motor(CRUISE_SPEED // 2, CRUISE_SPEED // 2)
            else:
                # Clear path — cruise forward
                await self._client.send_motor(CRUISE_SPEED, CRUISE_SPEED)

            await asyncio.sleep(0.1)  # ~10 Hz decision rate

    async def run(self) -> None:
        self._running = True
        self._load_model()

        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(self._on_telemetry)

        await self._client.connect()
        log.info("Vision engine running — autonomous mode active")

        try:
            await asyncio.gather(
                self._client.listen(),
                self._decision_loop(),
            )
        except KeyboardInterrupt:
            pass
        finally:
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
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    args = parser.parse_args()

    engine = VisionEngine(robot_url=args.robot, model=args.model)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
