"""Streaming dataset format helpers for autonomy recordings and replay."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO, cast

DATASET_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
FRAME_LOG_NAME = "frames.jsonl"
TELEMETRY_LOG_NAME = "telemetry.jsonl"


@dataclass(frozen=True)
class FrameRecord:
    index: int
    path: Path
    t_monotonic: float
    t_wall: float


@dataclass(frozen=True)
class TelemetryRecord:
    t_monotonic: float
    t_wall: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class DatasetEvent:
    channel: Literal["frame", "telemetry"]
    t_monotonic: float
    payload: FrameRecord | TelemetryRecord


@dataclass(frozen=True)
class FrameTick:
    frame: FrameRecord
    telemetry: TelemetryRecord | None


class DatasetWriter:
    def __init__(self, output_dir: Path, *, source: str, cli_args: dict[str, Any] | None = None) -> None:
        manifest = load_manifest(output_dir) if output_dir.exists() else None
        self._output_dir = output_dir
        self._source = source
        self._cli_args = cli_args or {}
        self._frame_log = output_dir / FRAME_LOG_NAME
        self._telemetry_log = output_dir / TELEMETRY_LOG_NAME
        self._manifest_path = output_dir / MANIFEST_NAME
        existing_counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
        self._frames_written = int(existing_counts.get("frames", 0))
        self._telemetry_written = int(existing_counts.get("telemetry", 0))
        self._started_monotonic = (
            float(manifest["started_monotonic"])
            if isinstance(manifest, dict) and manifest.get("started_monotonic") is not None
            else None
        )
        self._started_wall = (
            float(manifest["started_wall"])
            if isinstance(manifest, dict) and manifest.get("started_wall") is not None
            else None
        )
        self._frame_fp: TextIO | None = None
        self._telemetry_fp: TextIO | None = None

    def open(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_fp = self._frame_log.open("a", encoding="utf-8")
        self._telemetry_fp = self._telemetry_log.open("a", encoding="utf-8")
        self._write_manifest(status="recording")

    def write_frame(self, *, frame_index: int, t_monotonic: float, t_wall: float) -> Path:
        self._set_start_times(t_monotonic=t_monotonic, t_wall=t_wall)
        frame_path = self._output_dir / f"{frame_index:06d}.png"
        self._write_json_line(
            self._frame_fp,
            {
                "frame_index": frame_index,
                "filename": frame_path.name,
                "t_monotonic": t_monotonic,
                "t_wall": t_wall,
            },
        )
        self._frames_written += 1
        return frame_path

    def write_telemetry(self, payload: dict[str, Any], *, t_monotonic: float, t_wall: float) -> None:
        self._set_start_times(t_monotonic=t_monotonic, t_wall=t_wall)
        self._write_json_line(
            self._telemetry_fp,
            {
                "t_monotonic": t_monotonic,
                "t_wall": t_wall,
                "payload": payload,
            },
        )
        self._telemetry_written += 1

    def close(self, *, status: str = "complete") -> None:
        if self._frame_fp is not None:
            self._frame_fp.close()
            self._frame_fp = None
        if self._telemetry_fp is not None:
            self._telemetry_fp.close()
            self._telemetry_fp = None
        self._write_manifest(status=status)

    def _set_start_times(self, *, t_monotonic: float, t_wall: float) -> None:
        if self._started_monotonic is None:
            self._started_monotonic = t_monotonic
        if self._started_wall is None:
            self._started_wall = t_wall

    def _write_manifest(self, *, status: str) -> None:
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_kind": "autonomy_stream",
            "source": self._source,
            "status": status,
            "channels": {
                "frames": {
                    "image_glob": "*.png",
                    "metadata_log": FRAME_LOG_NAME,
                    "timestamp": "t_monotonic",
                },
                "telemetry": {
                    "jsonl": TELEMETRY_LOG_NAME,
                    "timestamp": "t_monotonic",
                    "payload": "robot websocket telemetry payload",
                },
            },
            "counts": {
                "frames": self._frames_written,
                "telemetry": self._telemetry_written,
            },
            "started_monotonic": self._started_monotonic,
            "started_wall": self._started_wall,
            "cli_args": self._cli_args,
        }
        with self._manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, sort_keys=True)
            fp.write("\n")

    @staticmethod
    def _write_json_line(fp: TextIO | None, payload: dict[str, Any]) -> None:
        if fp is None:
            msg = "dataset writer is not open"
            raise RuntimeError(msg)
        fp.write(json.dumps(payload, sort_keys=True))
        fp.write("\n")
        fp.flush()


def load_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def is_stream_dataset(dataset_dir: Path) -> bool:
    return (dataset_dir / MANIFEST_NAME).exists() or (dataset_dir / FRAME_LOG_NAME).exists()


def iter_frame_records(dataset_dir: Path) -> Iterator[FrameRecord]:
    if not dataset_dir.exists():
        return
    frame_log = dataset_dir / FRAME_LOG_NAME
    if frame_log.exists():
        with frame_log.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield FrameRecord(
                    index=int(row["frame_index"]),
                    path=dataset_dir / str(row["filename"]),
                    t_monotonic=float(row["t_monotonic"]),
                    t_wall=float(row["t_wall"]),
                )
        return

    frame_paths = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for idx, frame_path in enumerate(frame_paths):
        yield FrameRecord(
            index=idx,
            path=frame_path,
            t_monotonic=float(idx),
            t_wall=0.0,
        )


def iter_telemetry_records(dataset_dir: Path) -> Iterator[TelemetryRecord]:
    telemetry_path = dataset_dir / TELEMETRY_LOG_NAME
    if not telemetry_path.exists():
        return
    with telemetry_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"raw": payload}
            yield TelemetryRecord(
                t_monotonic=float(row["t_monotonic"]),
                t_wall=float(row["t_wall"]),
                payload=payload,
            )


def iter_dataset_events(dataset_dir: Path) -> Iterator[DatasetEvent]:
    frame_iter = iter_frame_records(dataset_dir)
    telemetry_iter = iter_telemetry_records(dataset_dir)

    next_frame = next(frame_iter, None)
    next_telemetry = next(telemetry_iter, None)

    while next_frame is not None or next_telemetry is not None:
        if next_telemetry is None or (next_frame is not None and next_frame.t_monotonic <= next_telemetry.t_monotonic):
            frame = next_frame
            if frame is None:
                break
            yield DatasetEvent(channel="frame", t_monotonic=frame.t_monotonic, payload=frame)
            next_frame = next(frame_iter, None)
            continue

        yield DatasetEvent(channel="telemetry", t_monotonic=next_telemetry.t_monotonic, payload=next_telemetry)
        next_telemetry = next(telemetry_iter, None)


def iter_frame_ticks(dataset_dir: Path) -> Iterator[FrameTick]:
    latest_telemetry: TelemetryRecord | None = None
    for event in iter_dataset_events(dataset_dir):
        if event.channel == "telemetry":
            if isinstance(event.payload, TelemetryRecord):
                latest_telemetry = event.payload
            continue
        if isinstance(event.payload, FrameRecord):
            yield FrameTick(frame=event.payload, telemetry=latest_telemetry)
