"""Picamera2 wrapper for JPEG streaming.

Provides frames as JPEG bytes via an asyncio-compatible interface.
"""

from __future__ import annotations

import asyncio
import io
import logging
from threading import Condition

log = logging.getLogger(__name__)


class _StreamBuffer(io.BufferedIOBase):
    """Thread-safe frame buffer written to by picamera2's encoder."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class Camera:
    def __init__(self, stream_size: tuple[int, int] = (400, 300), hflip: bool = False, vflip: bool = False) -> None:
        from picamera2 import Picamera2
        from libcamera import Transform

        self._cam = Picamera2()
        transform = Transform(hflip=int(hflip), vflip=int(vflip))

        preview_cfg = self._cam.create_preview_configuration(
            main={"size": (640, 480)}, transform=transform,
        )
        self._cam.configure(preview_cfg)

        self._stream_cfg = self._cam.create_video_configuration(
            main={"size": stream_size}, transform=transform,
        )
        self._buffer = _StreamBuffer()
        self._streaming = False
        self._stream_size = stream_size
        log.info("Camera initialised (%dx%d)", *stream_size)

    def start_stream(self) -> None:
        if self._streaming:
            return
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput

        if self._cam.started:
            self._cam.stop()
        self._cam.configure(self._stream_cfg)
        self._cam.start_recording(JpegEncoder(), FileOutput(self._buffer))
        self._streaming = True

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        self._cam.stop_recording()
        self._streaming = False

    def get_frame_sync(self, timeout: float = 1.0) -> bytes | None:
        """Blocking call — returns the next JPEG frame, or None on timeout."""
        with self._buffer.condition:
            if not self._buffer.condition.wait(timeout=timeout):
                return None
            return self._buffer.frame

    async def get_frame(self) -> bytes | None:
        """Async wrapper around the blocking frame wait."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_frame_sync)

    @property
    def streaming(self) -> bool:
        return self._streaming

    def close(self) -> None:
        self.stop_stream()
        self._cam.close()
