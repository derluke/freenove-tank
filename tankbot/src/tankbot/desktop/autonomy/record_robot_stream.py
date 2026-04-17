"""Record raw robot camera frames to a folder for vanilla MASt3R-SLAM runs."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tankbot.desktop.autonomy.dataset import DatasetWriter, iter_frame_records
from tankbot.desktop.autonomy.robot_client import RobotClient


class FrameRecorder:
    def __init__(self, robot_url: str, output_dir: Path, max_frames: int, duration_s: float) -> None:
        self._client = RobotClient(robot_url)
        self._output_dir = output_dir
        self._max_frames = max_frames
        self._duration_s = duration_s
        self._latest_frame: bytes | None = None
        self._latest_frame_monotonic = 0.0
        self._latest_frame_wall = 0.0
        self._written = 0
        self._start_time = 0.0
        self._dataset = DatasetWriter(
            output_dir,
            source="record_robot_stream",
            cli_args={
                "robot_url": robot_url,
                "max_frames": max_frames,
                "duration_s": duration_s,
            },
        )
        self._next_frame_index = self._discover_next_frame_index(output_dir)

    async def _on_frame(self, jpeg: bytes) -> None:
        self._latest_frame = jpeg
        self._latest_frame_monotonic = time.monotonic()
        self._latest_frame_wall = time.time()

    async def _on_telemetry(self, data: dict[str, Any]) -> None:
        self._dataset.write_telemetry(
            data,
            t_monotonic=time.monotonic(),
            t_wall=time.time(),
        )

    async def _on_imu(self, data: dict[str, Any]) -> None:
        self._dataset.write_imu(
            data,
            t_monotonic=time.monotonic(),
            t_wall=time.time(),
        )

    async def connect(self) -> None:
        self._dataset.open()
        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(self._on_telemetry)
        self._client.on_imu(self._on_imu)
        await self._client.connect()

    async def listen(self) -> None:
        await self._client.listen()

    async def record(self) -> int:
        self._start_time = time.monotonic()
        last_jpeg: bytes | None = None

        while True:
            if self._written >= self._max_frames:
                break
            if self._duration_s > 0 and (time.monotonic() - self._start_time) >= self._duration_s:
                break

            jpeg = self._latest_frame
            if jpeg is None or jpeg is last_jpeg:
                await asyncio.sleep(0.01)
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_path = self._output_dir / f"{self._next_frame_index:06d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                await asyncio.sleep(0.01)
                continue
            self._dataset.write_frame(
                frame_index=self._next_frame_index,
                t_monotonic=self._latest_frame_monotonic,
                t_wall=self._latest_frame_wall,
            )
            self._written += 1
            self._next_frame_index += 1
            last_jpeg = jpeg

        return self._written

    async def close(self) -> None:
        self._dataset.close()
        await self._client.close()

    @staticmethod
    def _discover_next_frame_index(output_dir: Path) -> int:
        next_index = 0
        for frame in iter_frame_records(output_dir):
            next_index = max(next_index, frame.index + 1)
        return next_index


async def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    recorder = FrameRecorder(
        robot_url=args.robot,
        output_dir=output_dir,
        max_frames=args.max_frames,
        duration_s=args.duration,
    )
    await recorder.connect()
    listener = asyncio.create_task(recorder.listen())
    try:
        written = await recorder.record()
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener
        await recorder.close()

    print(f"Recorded {written} frames to {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record robot frames plus raw telemetry to a streaming dataset folder."
    )
    parser.add_argument("--robot", default="ws://localhost:9000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds (0 = disabled)")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
