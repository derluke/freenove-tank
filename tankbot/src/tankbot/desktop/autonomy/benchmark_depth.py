"""Phase 0b depth-model benchmark.

Measures monocular depth candidates against the Phase 0b performance gate:

    Sustained end-to-end decision-loop rate: >=10 Hz floor, >=15 Hz target
    Worst-case per-frame latency:            <=120 ms (p99)
    Peak GPU memory:                         reported, compared against budget

Inputs come from either a streaming autonomy dataset (manifest.json +
frames.jsonl, see dataset.py) or a legacy PNG/JPG folder. The benchmark does
not hit the wire — it replays stored frames in a tight loop, which is
pessimistic for the control-loop rate (no camera or network latency is
amortised) but gives a clean per-model measurement.

Outputs:
  - per-backend JSON report under `logs/depth_bench/<timestamp>/<backend>.json`
  - per-backend colorized depth sample for quick visual sanity check
  - summary table printed to stdout, including gate pass/fail
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from tankbot.desktop.autonomy.dataset import iter_frame_records
from tankbot.desktop.autonomy.depth import (
    BackendUnavailable,
    DepthEstimator,
    DepthResult,
    build_estimator,
    list_backends,
)

log = logging.getLogger(__name__)

# Phase 0b gate (doc §3.1.1)
GATE_FPS_TARGET = 15.0
GATE_FPS_FLOOR = 10.0
GATE_P99_LATENCY_MS = 120.0


@dataclass
class BackendReport:
    backend: str
    status: str  # "ok" | "unavailable" | "error"
    detail: str = ""
    warmup_count: int = 0
    measured_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    peak_gpu_mem_mib: float | None = None
    sample_depth_stats: dict[str, float] | None = None

    def summary(self) -> dict[str, object]:
        if not self.latencies_ms:
            return {
                "backend": self.backend,
                "status": self.status,
                "detail": self.detail,
            }
        lats = sorted(self.latencies_ms)
        n = len(lats)

        def _quantile(q: float) -> float:
            return lats[min(n - 1, int(q * n))]

        mean_ms = mean(lats)
        sustained_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        p50 = median(lats)
        p95 = _quantile(0.95)
        p99 = _quantile(0.99)
        gate_fps_pass = sustained_fps >= GATE_FPS_TARGET
        gate_floor_pass = sustained_fps >= GATE_FPS_FLOOR
        gate_latency_pass = p99 <= GATE_P99_LATENCY_MS
        overall = (
            "PASS"
            if gate_fps_pass and gate_latency_pass
            else ("FLOOR" if gate_floor_pass and gate_latency_pass else "FAIL")
        )
        return {
            "backend": self.backend,
            "status": self.status,
            "detail": self.detail,
            "measured_frames": n,
            "warmup_frames": self.warmup_count,
            "latency_ms": {
                "mean": round(mean_ms, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "min": round(min(lats), 2),
                "max": round(max(lats), 2),
            },
            "sustained_fps": round(sustained_fps, 2),
            "peak_gpu_mem_mib": round(self.peak_gpu_mem_mib, 1) if self.peak_gpu_mem_mib else None,
            "sample_depth_stats": self.sample_depth_stats,
            "gate": {
                "target_fps": GATE_FPS_TARGET,
                "floor_fps": GATE_FPS_FLOOR,
                "p99_ms_max": GATE_P99_LATENCY_MS,
                "fps_target_pass": gate_fps_pass,
                "fps_floor_pass": gate_floor_pass,
                "p99_pass": gate_latency_pass,
                "verdict": overall,
            },
        }


def _load_frames(dataset_dir: Path, max_frames: int) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for record in iter_frame_records(dataset_dir):
        if len(images) >= max_frames:
            break
        img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("Skipping unreadable frame %s", record.path)
            continue
        images.append(img)
    if not images:
        msg = f"No frames loaded from {dataset_dir}"
        raise SystemExit(msg)
    return images


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(finite, (2, 98))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _read_gpu_peak_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _stats_for_sample(result: DepthResult) -> dict[str, float]:
    depth = result.depth_m
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "valid_ratio": 0.0}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "valid_ratio": float(finite.size / depth.size),
    }


def _benchmark_backend(
    estimator: DepthEstimator,
    frames: list[np.ndarray],
    warmup: int,
    measured: int,
    output_dir: Path,
) -> BackendReport:
    report = BackendReport(backend=estimator.name, status="ok")

    try:
        estimator.load()
    except BackendUnavailable as exc:
        log.warning("%s unavailable: %s", estimator.name, exc)
        report.status = "unavailable"
        report.detail = str(exc)
        return report
    except Exception as exc:
        log.exception("%s failed to load", estimator.name)
        report.status = "error"
        report.detail = f"load: {exc}"
        return report

    try:
        # Warmup — first inferences are slow (cudnn autotune, lazy kernels).
        _reset_gpu_peak()
        for i in range(warmup):
            estimator.infer(frames[i % len(frames)])
            report.warmup_count += 1
        _cuda_sync()

        # Measurement loop.
        _reset_gpu_peak()
        last_result: DepthResult | None = None
        for i in range(measured):
            frame = frames[i % len(frames)]
            t0 = time.perf_counter()
            result = estimator.infer(frame)
            _cuda_sync()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            report.latencies_ms.append(elapsed_ms)
            report.measured_count += 1
            last_result = result

        report.peak_gpu_mem_mib = _read_gpu_peak_mib()

        if last_result is not None:
            report.sample_depth_stats = _stats_for_sample(last_result)
            sample_path = output_dir / f"{estimator.name}_sample.png"
            cv2.imwrite(str(sample_path), _colorize_depth(last_result.depth_m))

    except Exception as exc:
        log.exception("%s failed during inference", estimator.name)
        report.status = "error"
        report.detail = f"infer: {exc}"
    finally:
        estimator.unload()

    return report


def _print_summary(summaries: list[dict[str, object]]) -> None:
    print()
    print(
        f"{'backend':24s}  {'status':12s}  {'mean ms':>9s}  {'p99 ms':>9s}  {'fps':>7s}  {'mem MiB':>9s}  {'verdict':>8s}"
    )
    print("-" * 90)
    for row in summaries:
        status = str(row.get("status", "?"))
        if status != "ok":
            print(f"{row['backend']:24s}  {status:12s}  {'-':>9s}  {'-':>9s}  {'-':>7s}  {'-':>9s}  {'-':>8s}")
            continue
        lat = row.get("latency_ms", {}) or {}
        mean_ms = lat.get("mean", 0.0)
        p99_ms = lat.get("p99", 0.0)
        fps = row.get("sustained_fps", 0.0)
        mem = row.get("peak_gpu_mem_mib") or 0.0
        verdict = (row.get("gate", {}) or {}).get("verdict", "?")
        print(
            f"{row['backend']:24s}  {status:12s}  {mean_ms:9.2f}  {p99_ms:9.2f}  {fps:7.2f}  {mem:9.1f}  {verdict:>8s}"
        )
    print()
    print(
        f"Phase 0b gate — target: >={GATE_FPS_TARGET:.0f} Hz sustained, "
        f"floor: >={GATE_FPS_FLOOR:.0f} Hz, p99 <={GATE_P99_LATENCY_MS:.0f} ms"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0b monocular depth benchmark against the performance gate.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Streaming autonomy dataset or plain PNG/JPG folder of replay frames.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list_backends(),
        help=f"Backends to run (default: all). Known: {', '.join(list_backends())}",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per backend.")
    parser.add_argument("--measured", type=int, default=120, help="Measured iterations per backend.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Cap on distinct frames loaded from the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to write per-backend reports. Defaults to logs/depth_bench/<timestamp>/",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _build_parser().parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        log.error("Dataset directory not found: %s", dataset_dir)
        return 2

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path("logs") / "depth_bench" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Reports will be written to %s", output_dir)

    frames = _load_frames(dataset_dir, args.max_frames)
    log.info(
        "Loaded %d frames from %s (resolution %dx%d)", len(frames), dataset_dir, frames[0].shape[1], frames[0].shape[0]
    )

    summaries: list[dict[str, object]] = []
    for backend_name in args.backends:
        if backend_name not in list_backends():
            log.warning("Unknown backend %s — skipping", backend_name)
            continue
        log.info("Benchmarking %s ...", backend_name)
        estimator = build_estimator(backend_name)
        report = _benchmark_backend(
            estimator, frames, warmup=args.warmup, measured=args.measured, output_dir=output_dir
        )
        summary = report.summary()
        summaries.append(summary)
        report_path = output_dir / f"{backend_name}.json"
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log.info("Wrote %s", report_path)

    overall_path = output_dir / "summary.json"
    overall_path.write_text(
        json.dumps(
            {
                "gate": {
                    "target_fps": GATE_FPS_TARGET,
                    "floor_fps": GATE_FPS_FLOOR,
                    "p99_ms_max": GATE_P99_LATENCY_MS,
                },
                "dataset": str(dataset_dir),
                "frames_loaded": len(frames),
                "warmup": args.warmup,
                "measured": args.measured,
                "backends": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %s", overall_path)

    _print_summary(summaries)
    # Non-zero exit if nothing passed the target gate — helps CI-style usage.
    any_pass = any((s.get("status") == "ok" and (s.get("gate", {}) or {}).get("verdict") == "PASS") for s in summaries)
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
