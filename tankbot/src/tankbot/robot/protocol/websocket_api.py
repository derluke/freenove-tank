"""WebSocket API server for modern clients (Phoenix app, autonomy engine).

Provides:
  - JSON command/telemetry on text frames
  - Binary JPEG video frames
  - Event subscription model
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

WS_PORT = 9000


class WebSocketAPI:
    """Lightweight WebSocket server using the `websockets` library."""

    def __init__(self, on_command: Callable[[dict], Awaitable[None] | None]) -> None:
        self._on_command = on_command
        self._clients: set = set()
        self._video_clients: set = set()
        self._server = None

    async def start(self, host: str, port: int = WS_PORT) -> None:
        import websockets

        self._server = await websockets.serve(self._handle_client, host, port)
        log.info("WebSocket API listening on ws://%s:%d", host, port)

    async def _handle_client(self, websocket) -> None:
        self._clients.add(websocket)
        addr = websocket.remote_address
        log.info("WS client connected: %s", addr)

        try:
            async for raw in websocket:
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    # Handle subscription
                    if msg.get("type") == "subscribe" and "video" in msg.get("channels", []):
                        self._video_clients.add(websocket)
                        continue
                    result = self._on_command(msg)
                    if asyncio.iscoroutine(result):
                        await result
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            self._video_clients.discard(websocket)
            log.info("WS client disconnected: %s", addr)

    async def broadcast_telemetry(self, data: dict) -> None:
        if not self._clients:
            return
        payload = json.dumps(data)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
            self._video_clients.discard(ws)

    async def broadcast_frame(self, jpeg_data: bytes) -> None:
        if not self._video_clients:
            return
        dead = []
        for ws in list(self._video_clients):
            try:
                await ws.send(jpeg_data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._video_clients.discard(ws)
            self._clients.discard(ws)

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    @property
    def has_video_clients(self) -> bool:
        return len(self._video_clients) > 0

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
