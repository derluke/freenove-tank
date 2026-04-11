"""Freenove-compatible TCP servers (command port 5003, video port 8003).

This is a drop-in replacement for the original server.py / tcp_server.py
using asyncio instead of threads + select.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from tankbot.shared.protocol import LEGACY_CMD_PORT, LEGACY_FRAME_HEADER, LEGACY_VIDEO_PORT

log = logging.getLogger(__name__)


class LegacyCmdServer:
    """Async TCP server on port 5003 — handles text command protocol."""

    def __init__(self, on_message: Callable[[str], Awaitable[None] | None]) -> None:
        self._on_message = on_message
        self._server: asyncio.Server | None = None
        self._clients: dict[asyncio.StreamWriter, str] = {}  # writer → addr string

    async def start(self, host: str, port: int = LEGACY_CMD_PORT) -> None:
        self._server = await asyncio.start_server(self._handle_client, host, port)
        log.info("Legacy command server listening on %s:%d", host, port)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        addr_str = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        self._clients[writer] = addr_str
        log.info("Cmd client connected: %s", addr_str)

        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace").strip()
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        result = self._on_message(line)
                        if asyncio.iscoroutine(result):
                            await result
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._clients.pop(writer, None)
            writer.close()
            log.info("Cmd client disconnected: %s", addr_str)

    async def send_to_all(self, message: str) -> None:
        data = message.encode("utf-8")
        for writer in list(self._clients):
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                self._clients.pop(writer, None)
                writer.close()

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    async def stop(self) -> None:
        for writer in list(self._clients):
            writer.close()
        self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


class LegacyVideoServer:
    """Async TCP server on port 8003 — streams JPEG frames with 4-byte length prefix."""

    def __init__(self) -> None:
        self._server: asyncio.Server | None = None
        self._clients: dict[asyncio.StreamWriter, str] = {}

    async def start(self, host: str, port: int = LEGACY_VIDEO_PORT) -> None:
        self._server = await asyncio.start_server(self._handle_client, host, port)
        log.info("Legacy video server listening on %s:%d", host, port)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        addr_str = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        self._clients[writer] = addr_str
        log.info("Video client connected: %s", addr_str)
        # Keep connection open — we write frames, client just reads
        try:
            while True:
                data = await reader.read(1)
                if not data:
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._clients.pop(writer, None)
            writer.close()
            log.info("Video client disconnected: %s", addr_str)

    async def send_frame(self, jpeg_data: bytes) -> None:
        header = LEGACY_FRAME_HEADER.pack(len(jpeg_data))
        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._clients):
            try:
                writer.write(header + jpeg_data)
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, ConnectionError):
                dead.append(writer)
        for writer in dead:
            self._clients.pop(writer, None)
            writer.close()

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    async def stop(self) -> None:
        for writer in list(self._clients):
            writer.close()
        self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
