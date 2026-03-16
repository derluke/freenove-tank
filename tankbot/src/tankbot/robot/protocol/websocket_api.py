"""WebSocket API server for modern clients (Phoenix app, autonomy engine).

Provides:
  - JSON command/telemetry on text frames
  - Binary JPEG video frames
  - Event subscription model

Each client gets a dedicated send queue so that broadcasting telemetry/video
never blocks the receive loop (websockets requires serialised writes).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any

log = logging.getLogger(__name__)

WS_PORT = 9000


@dataclass
class _Client:
    ws: Any
    addr: str
    send_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=60))
    wants_video: bool = False


class WebSocketAPI:
    """Lightweight WebSocket server using the `websockets` library."""

    def __init__(self, on_command: Callable[[dict], Awaitable[None] | None]) -> None:
        self._on_command = on_command
        self._clients: dict[Any, _Client] = {}  # ws → _Client
        self._server = None

    async def start(self, host: str, port: int = WS_PORT) -> None:
        import websockets

        self._server = await websockets.serve(
            self._handle_client, host, port,
            ping_interval=None,
            max_size=2**22,
        )
        log.info("WebSocket API listening on ws://%s:%d", host, port)

    async def _handle_client(self, websocket) -> None:
        addr = websocket.remote_address
        addr_str = f"{addr}" if addr else "unknown"
        client = _Client(ws=websocket, addr=addr_str)
        self._clients[websocket] = client
        log.info("WS client connected: %s", addr_str)

        # Spawn a dedicated sender task for this client
        sender_task = asyncio.create_task(self._sender(client))

        try:
            async for raw in websocket:
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "subscribe" and "video" in msg.get("channels", []):
                        client.wants_video = True
                        log.info("WS client %s subscribed to video", addr_str)
                        continue
                    try:
                        result = self._on_command(msg)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as cmd_err:
                        log.warning("Command handler error: %s (msg=%s)", cmd_err, msg)
        except Exception as e:
            log.warning("WS client error (%s): %s", addr_str, e)
        finally:
            sender_task.cancel()
            self._clients.pop(websocket, None)
            log.info("WS client disconnected: %s", addr_str)

    async def _sender(self, client: _Client) -> None:
        """Dedicated send loop — drains the queue and writes to the websocket."""
        try:
            while True:
                data = await client.send_queue.get()
                await client.ws.send(data)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _enqueue(self, client: _Client, data: str | bytes) -> None:
        """Non-blocking enqueue — drops if the queue is full (backpressure)."""
        try:
            client.send_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass  # Drop frame/telemetry rather than block

    async def broadcast_telemetry(self, data: dict) -> None:
        if not self._clients:
            return
        payload = json.dumps(data)
        for client in list(self._clients.values()):
            self._enqueue(client, payload)

    async def broadcast_frame(self, jpeg_data: bytes) -> None:
        for client in list(self._clients.values()):
            if client.wants_video:
                self._enqueue(client, jpeg_data)

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    @property
    def has_video_clients(self) -> bool:
        return any(c.wants_video for c in self._clients.values())

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
