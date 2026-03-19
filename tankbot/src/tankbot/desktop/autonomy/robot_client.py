"""WebSocket client that connects to the robot and provides a clean API.

Used by the autonomy engine and Phoenix app (via ports) to control the robot.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import websockets

log = logging.getLogger(__name__)


class RobotClient:
    def __init__(self, robot_url: str = "") -> None:
        import os
        robot_url = robot_url or os.environ.get("ROBOT_URL", "ws://localhost:9000")
        self._url = robot_url
        self._ws: Any = None
        self._on_telemetry: Callable[[dict], Awaitable[None] | None] | None = None
        self._on_frame: Callable[[bytes], Awaitable[None] | None] | None = None
        self._running = False

    def on_telemetry(self, callback: Callable[[dict], Awaitable[None] | None]) -> None:
        self._on_telemetry = callback

    def on_frame(self, callback: Callable[[bytes], Awaitable[None] | None]) -> None:
        self._on_frame = callback

    async def connect(self) -> None:
        self._ws = await websockets.connect(self._url)
        self._running = True
        # Subscribe to video frames
        await self._ws.send(json.dumps({"type": "subscribe", "channels": ["video"]}))
        log.info("Connected to robot at %s", self._url)

    async def listen(self) -> None:
        """Process incoming messages from the robot."""
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes) and self._on_frame:
                    result = self._on_frame(msg)
                    if asyncio.iscoroutine(result):
                        await result
                elif isinstance(msg, str) and self._on_telemetry:
                    try:
                        data = json.loads(msg)
                        result = self._on_telemetry(data)
                        if asyncio.iscoroutine(result):
                            await result
                    except json.JSONDecodeError:
                        pass
        except websockets.ConnectionClosed:
            log.warning("Robot connection closed")
        finally:
            self._running = False

    async def send_motor(self, left: int, right: int) -> None:
        await self._send({"cmd": "motor", "left": left, "right": right})

    async def send_servo(self, channel: int, angle: int) -> None:
        await self._send({"cmd": "servo", "channel": channel, "angle": angle})

    async def send_led(self, r: int, g: int, b: int, mask: int = 0xF) -> None:
        await self._send({"cmd": "led", "r": r, "g": g, "b": b, "mask": mask})

    async def send_led_off(self) -> None:
        await self._send({"cmd": "led_off"})

    async def send_stop(self) -> None:
        await self._send({"cmd": "stop"})

    async def send_mode(self, mode: int) -> None:
        await self._send({"cmd": "mode", "mode": mode})

    async def send_vision_status(
        self,
        state: str,
        detections: list[dict],
        distance: float,
        depth_strips: tuple[float, float, float] | None = None,
        slam_data: dict | None = None,
        autonomy_data: dict | None = None,
    ) -> None:
        """Push vision engine state to the robot for dashboard display."""
        msg: dict = {
            "cmd": "vision",
            "state": state,
            "detections": detections,
            "distance": distance,
        }
        if depth_strips is not None:
            msg["depth"] = {"left": depth_strips[0], "center": depth_strips[1], "right": depth_strips[2]}
        if slam_data is not None:
            msg["slam"] = slam_data
        if autonomy_data is not None:
            msg["autonomy"] = autonomy_data
        await self._send(msg)

    async def _send(self, data: dict) -> None:
        if self._ws:
            await self._ws.send(json.dumps(data))

    async def close(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
