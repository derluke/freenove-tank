"""Offline scan-match evaluator + config sweeper.

Runs the Phase 2 scan-match pipeline against a recorded dataset with
configurable ``ScanMatchConfig``/``ICPConfig``/``ScanConfig`` overrides
and reports metrics relative to a tape-measured ground truth. Depth
inference is cached to a sidecar ``.npz`` so a config sweep runs
hundreds of combinations without re-loading the depth model.

Subcommands:
- ``eval`` — run one config, print the metrics summary.
- ``sweep`` — grid-search a small set of knobs and print a ranked table.

The sweep is deliberately narrow: the goal here is to validate that
Option A (loosened residual/correspondence thresholds for monocular
depth noise) works on the ``straight_2m`` recording before making it
the default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tankbot.desktop.autonomy.dataset import (
    FrameRecord,
    ImuRecord,
    TelemetryRecord,
    iter_dataset_events,
)
from tankbot.desktop.autonomy.depth import build_estimator
from tankbot.desktop.autonomy.reactive import load_extrinsics, load_intrinsics
from tankbot.desktop.autonomy.reactive.gyro import GyroIntegrator
from tankbot.desktop.autonomy.reactive.pose import HealthState
from tankbot.desktop.autonomy.reactive.projection import (
    ProjectionConfig,
    project_depth_to_robot_points,
)
from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)

_DEFAULT_DEPTH_BACKEND = "depth_anything_v2"
_DEFAULT_EXTRINSICS = "calibration/camera_extrinsics.yaml"
_DEFAULT_INTRINSICS = "calibration/camera_intrinsics.yaml"
_DEFAULT_DEPTH_SCALE = 1.0


# ---------------------------------------------------------------------------
# Observations cache
# ---------------------------------------------------------------------------


@dataclass
class Observations:
    """Per-frame inputs for the scan-match estimator, cached to disk."""

    t_monotonic: np.ndarray  # (N,) float64
    frame_idx: np.ndarray  # (N,) int32
    gyro_yaw: np.ndarray  # (N,) float64 — rad, NaN before IMU start
    motors_active: np.ndarray  # (N,) bool
    last_motor_active_t: np.ndarray  # (N,) float64 — NaN when unseen
    points_offsets: np.ndarray  # (N+1,) int32 — CSR-style concat offsets
    points_xyz: np.ndarray  # (sum, 3) float32 robot-frame

    @property
    def n_frames(self) -> int:
        return int(self.t_monotonic.shape[0])

    def frame_points(self, i: int) -> np.ndarray:
        start = int(self.points_offsets[i])
        end = int(self.points_offsets[i + 1])
        return self.points_xyz[start:end]


def _cache_path(dataset: Path, backend: str, proj_cfg: ProjectionConfig) -> Path:
    return dataset / f"observations_{backend}_{proj_cfg.cache_tag()}.npz"


def _save_observations(path: Path, obs: Observations) -> None:
    np.savez_compressed(
        path,
        t_monotonic=obs.t_monotonic,
        frame_idx=obs.frame_idx,
        gyro_yaw=obs.gyro_yaw,
        motors_active=obs.motors_active,
        last_motor_active_t=obs.last_motor_active_t,
        points_offsets=obs.points_offsets,
        points_xyz=obs.points_xyz,
    )


def _derive_motor_activity_sidechannel(dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    """Frame-aligned motor state plus exact last-nonzero timestamp."""
    motors_active = False
    motor_seen = False
    last_motor_active_t: float | None = None
    active_rows: list[bool] = []
    last_active_rows: list[float] = []

    for event in iter_dataset_events(dataset):
        payload = event.payload
        if event.channel == "imu" and isinstance(payload, ImuRecord):
            if payload.motor_left is not None and payload.motor_right is not None:
                motors_active = payload.motor_left != 0 or payload.motor_right != 0
                motor_seen = True
                if motors_active:
                    last_motor_active_t = payload.t_monotonic
        elif event.channel == "telemetry" and isinstance(payload, TelemetryRecord):
            motor = payload.payload.get("motor")
            if isinstance(motor, dict):
                left = int(motor.get("left", 0))
                right = int(motor.get("right", 0))
                motors_active = left != 0 or right != 0
                motor_seen = True
                if motors_active:
                    last_motor_active_t = payload.t_monotonic
        elif event.channel == "frame" and isinstance(payload, FrameRecord):
            active_rows.append(motors_active if motor_seen else True)
            last_active_rows.append(last_motor_active_t if last_motor_active_t is not None else float("nan"))

    return (
        np.asarray(active_rows, dtype=bool),
        np.asarray(last_active_rows, dtype=np.float64),
    )


def _load_observations(path: Path) -> Observations:
    data = np.load(path)
    if "last_motor_active_t" in data:
        last_motor_active_t = data["last_motor_active_t"]
    else:
        # Older caches only stored the sampled motor state at frame time.
        # Rehydrate the exact last-nonzero event timestamp from the dataset
        # logs so replay can honor pulse/coast gaps without re-running depth.
        _motors_active, last_motor_active_t = _derive_motor_activity_sidechannel(path.parent)
        if len(_motors_active) == data["t_monotonic"].shape[0]:
            motors_active = _motors_active
        else:
            motors_active = data["motors_active"].astype(bool)
            last_motor_active_t = np.full(data["t_monotonic"].shape[0], float("nan"), dtype=np.float64)
    if "last_motor_active_t" in data:
        motors_active = data["motors_active"].astype(bool)
    return Observations(
        t_monotonic=data["t_monotonic"],
        frame_idx=data["frame_idx"].astype(np.int32),
        gyro_yaw=data["gyro_yaw"],
        motors_active=motors_active,
        last_motor_active_t=np.asarray(last_motor_active_t, dtype=np.float64),
        points_offsets=data["points_offsets"].astype(np.int32),
        points_xyz=data["points_xyz"].astype(np.float32),
    )


def _slice_observations(obs: Observations, max_frames: int | None) -> Observations:
    if max_frames is None or max_frames >= obs.n_frames:
        return obs
    n = max(0, int(max_frames))
    end_offset = int(obs.points_offsets[n])
    return Observations(
        t_monotonic=obs.t_monotonic[:n].copy(),
        frame_idx=obs.frame_idx[:n].copy(),
        gyro_yaw=obs.gyro_yaw[:n].copy(),
        motors_active=obs.motors_active[:n].copy(),
        last_motor_active_t=obs.last_motor_active_t[:n].copy(),
        points_offsets=obs.points_offsets[: n + 1].copy(),
        points_xyz=obs.points_xyz[:end_offset].copy(),
    )


def build_observations(
    dataset: Path,
    *,
    backend: str,
    depth_scale: float,
    intrinsics_path: Path,
    extrinsics_path: Path,
    max_frames: int | None,
    force: bool = False,
    verbose: bool = True,
) -> Observations:
    """Build (or reload) the per-frame observations cache for ``dataset``."""
    proj_cfg = ProjectionConfig(depth_scale=depth_scale)
    cache = _cache_path(dataset, backend, proj_cfg)
    if cache.exists() and not force:
        if verbose:
            print(f"Loading cached observations: {cache.name}")
        return _slice_observations(_load_observations(cache), max_frames)

    import cv2

    intrinsics = load_intrinsics(intrinsics_path)
    extrinsics = load_extrinsics(extrinsics_path)

    if verbose:
        print(f"Loading depth backend: {backend} ...")
    estimator = build_estimator(backend)
    estimator.load()

    gyro = GyroIntegrator()
    motors_active = False
    motor_seen = False
    imu_seen = False

    ts: list[float] = []
    idxs: list[int] = []
    yaws: list[float] = []
    mots: list[bool] = []
    motor_active_ts: list[float] = []
    pts_chunks: list[np.ndarray] = []
    offsets: list[int] = [0]
    running = 0
    last_motor_active_t: float | None = None

    frames_done = 0
    try:
        for event in iter_dataset_events(dataset):
            payload = event.payload
            if event.channel == "imu" and isinstance(payload, ImuRecord):
                gyro.add_sample(payload.gz, t=payload.t_monotonic)
                imu_seen = True
                if payload.motor_left is not None and payload.motor_right is not None:
                    motors_active = payload.motor_left != 0 or payload.motor_right != 0
                    motor_seen = True
                    if motors_active:
                        last_motor_active_t = payload.t_monotonic
            elif event.channel == "telemetry" and isinstance(payload, TelemetryRecord):
                motor = payload.payload.get("motor")
                if isinstance(motor, dict):
                    left = int(motor.get("left", 0))
                    right = int(motor.get("right", 0))
                    motors_active = left != 0 or right != 0
                    motor_seen = True
                    if motors_active:
                        last_motor_active_t = payload.t_monotonic
            elif event.channel == "frame" and isinstance(payload, FrameRecord):
                bgr = cv2.imread(str(payload.path))
                if bgr is None:
                    continue
                t0 = time.monotonic()
                depth_result = estimator.infer(bgr)
                points = project_depth_to_robot_points(
                    depth_result.depth_m,
                    intrinsics,
                    extrinsics,
                    proj_cfg,
                ).astype(np.float32)

                ts.append(payload.t_monotonic)
                idxs.append(payload.index)
                yaws.append(gyro.yaw_rad if imu_seen else float("nan"))
                mots.append(motors_active if motor_seen else True)
                motor_active_ts.append(last_motor_active_t if last_motor_active_t is not None else float("nan"))
                pts_chunks.append(points)
                running += points.shape[0]
                offsets.append(running)

                frames_done += 1
                if verbose and frames_done % 100 == 0:
                    dt = (time.monotonic() - t0) * 1000.0
                    print(f"  frame {frames_done} ({dt:.0f} ms) pts={points.shape[0]}")

                if max_frames is not None and frames_done >= max_frames:
                    break
    finally:
        estimator.unload()

    if frames_done == 0:
        msg = f"No frames readable in {dataset}"
        raise RuntimeError(msg)

    if not pts_chunks:
        points_xyz = np.zeros((0, 3), dtype=np.float32)
    else:
        points_xyz = np.concatenate(pts_chunks, axis=0).astype(np.float32)

    obs = Observations(
        t_monotonic=np.asarray(ts, dtype=np.float64),
        frame_idx=np.asarray(idxs, dtype=np.int32),
        gyro_yaw=np.asarray(yaws, dtype=np.float64),
        motors_active=np.asarray(mots, dtype=bool),
        last_motor_active_t=np.asarray(motor_active_ts, dtype=np.float64),
        points_offsets=np.asarray(offsets, dtype=np.int32),
        points_xyz=points_xyz,
    )

    # Only cache a full-dataset run. Partial runs would poison future loads.
    if max_frames is None:
        if verbose:
            print(f"Caching observations to {cache.name} ({points_xyz.shape[0]} points)")
        _save_observations(cache, obs)

    return obs


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMetrics:
    """Metrics emitted per config run. All fields printable in one table row."""

    n_frames: int
    final_xy: tuple[float, float] | None
    last_valid_xy: tuple[float, float] | None
    max_x_reached: float
    final_yaw_deg: float
    endpoint_error_m: float
    heading_error_deg: float
    frac_healthy: float
    frac_degraded: float
    frac_broken: float
    frac_xy_valid: float
    residual_p50: float
    residual_p95: float
    residual_inf_rate: float
    mean_corr: float

    def as_row(self) -> dict[str, Any]:
        fx, fy = self.final_xy if self.final_xy is not None else (float("nan"), float("nan"))
        lvx, lvy = self.last_valid_xy if self.last_valid_xy is not None else (float("nan"), float("nan"))
        return {
            "n_frames": self.n_frames,
            "final_x": fx,
            "final_y": fy,
            "last_valid_x": lvx,
            "last_valid_y": lvy,
            "max_x_reached": self.max_x_reached,
            "final_yaw_deg": self.final_yaw_deg,
            "endpoint_err_m": self.endpoint_error_m,
            "heading_err_deg": self.heading_error_deg,
            "frac_healthy": self.frac_healthy,
            "frac_degraded": self.frac_degraded,
            "frac_broken": self.frac_broken,
            "frac_xy_valid": self.frac_xy_valid,
            "residual_p50": self.residual_p50,
            "residual_p95": self.residual_p95,
            "residual_inf_rate": self.residual_inf_rate,
            "mean_corr": self.mean_corr,
        }


@dataclass(frozen=True)
class GroundTruth:
    start_xy_m: tuple[float, float]
    start_yaw_deg: float
    end_xy_m: tuple[float, float]
    end_yaw_deg: float

    @classmethod
    def from_json(cls, path: Path) -> GroundTruth:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            start_xy_m=(float(data["start_xy_m"][0]), float(data["start_xy_m"][1])),
            start_yaw_deg=float(data.get("start_yaw_deg", 0.0)),
            end_xy_m=(float(data["end_xy_m"][0]), float(data["end_xy_m"][1])),
            end_yaw_deg=float(data.get("end_yaw_deg", 0.0)),
        )


def evaluate_run(
    obs: Observations,
    *,
    scan_cfg: ScanConfig,
    icp_cfg: ICPConfig,
    sm_cfg: ScanMatchConfig,
    ground_truth: GroundTruth,
) -> EvalMetrics:
    """Replay ``obs`` through ScanMatchPoseSource with the given configs."""
    pose_source = ScanMatchPoseSource(
        scan_config=scan_cfg,
        icp_config=icp_cfg,
        config=sm_cfg,
    )

    residuals: list[float] = []
    n_inf = 0
    corr_counts: list[int] = []
    health_counts = {HealthState.HEALTHY: 0, HealthState.DEGRADED: 0, HealthState.BROKEN: 0}
    xy_valid = 0
    last_valid_xy: tuple[float, float] | None = None
    max_x_reached = 0.0

    for i in range(obs.n_frames):
        points = obs.frame_points(i)
        gyro_yaw = float(obs.gyro_yaw[i])
        gyro_arg = None if math.isnan(gyro_yaw) else gyro_yaw
        last_motor_active_t = float(obs.last_motor_active_t[i])
        last_motor_active_arg = None if math.isnan(last_motor_active_t) else last_motor_active_t
        pose_source.update(
            points,
            gyro_yaw=gyro_arg,
            motors_active=bool(obs.motors_active[i]),
            last_motor_active_t=last_motor_active_arg,
            t_monotonic=float(obs.t_monotonic[i]),
        )

        pose = pose_source.latest()
        health_counts[pose.health] = health_counts.get(pose.health, 0) + 1
        if pose.xy_m is not None:
            xy_valid += 1
            last_valid_xy = pose.xy_m
            if pose.xy_m[0] > max_x_reached:
                max_x_reached = pose.xy_m[0]

        match = pose_source.last_match
        if match is not None:
            if math.isinf(match.mean_residual):
                n_inf += 1
            else:
                residuals.append(match.mean_residual)
            corr_counts.append(match.n_correspondences)

    pose = pose_source.latest()
    final_xy = pose.xy_m
    final_yaw_rad = pose.yaw_rad

    # Prefer final xy, fall back to last-known-valid for the metric.
    xy_for_metric = final_xy if final_xy is not None else last_valid_xy
    if xy_for_metric is not None:
        ex = xy_for_metric[0] - ground_truth.end_xy_m[0]
        ey = xy_for_metric[1] - ground_truth.end_xy_m[1]
        endpoint_err = math.hypot(ex, ey)
    else:
        endpoint_err = float("nan")

    heading_err = math.degrees(final_yaw_rad) - ground_truth.end_yaw_deg

    n = max(obs.n_frames, 1)
    return EvalMetrics(
        n_frames=obs.n_frames,
        final_xy=final_xy,
        last_valid_xy=last_valid_xy,
        max_x_reached=max_x_reached,
        final_yaw_deg=math.degrees(final_yaw_rad),
        endpoint_error_m=endpoint_err,
        heading_error_deg=heading_err,
        frac_healthy=health_counts[HealthState.HEALTHY] / n,
        frac_degraded=health_counts[HealthState.DEGRADED] / n,
        frac_broken=health_counts[HealthState.BROKEN] / n,
        frac_xy_valid=xy_valid / n,
        residual_p50=float(np.median(residuals)) if residuals else float("nan"),
        residual_p95=float(np.percentile(residuals, 95)) if residuals else float("nan"),
        residual_inf_rate=n_inf / max(len(residuals) + n_inf, 1),
        mean_corr=float(np.mean(corr_counts)) if corr_counts else float("nan"),
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _print_metrics(m: EvalMetrics, label: str = "") -> None:
    if label:
        print(f"\n=== {label} ===")
    fx, fy = m.final_xy if m.final_xy is not None else (float("nan"), float("nan"))
    lvx, lvy = m.last_valid_xy if m.last_valid_xy is not None else (float("nan"), float("nan"))
    print(f"  frames:            {m.n_frames}")
    print(f"  final pose:        ({fx:+.3f}, {fy:+.3f}) m  yaw={m.final_yaw_deg:+.2f} deg")
    print(f"  last valid xy:     ({lvx:+.3f}, {lvy:+.3f}) m   max x: {m.max_x_reached:+.3f} m")
    print(f"  endpoint error:    {m.endpoint_error_m:.3f} m")
    print(f"  heading error:     {m.heading_error_deg:+.2f} deg")
    print(f"  health:            healthy={m.frac_healthy:.0%}  degraded={m.frac_degraded:.0%}  broken={m.frac_broken:.0%}")
    print(f"  xy_m valid:        {m.frac_xy_valid:.0%}")
    print(f"  residual p50/p95:  {m.residual_p50:.3f} / {m.residual_p95:.3f} m")
    print(f"  residual inf rate: {m.residual_inf_rate:.0%}")
    print(f"  mean corr:         {m.mean_corr:.1f}")


def _score(m: EvalMetrics, end_x_target: float) -> float:
    """Lower is better.

    Favors configs that:
    - track endpoint close to tape-measured target (using last-valid xy)
    - spend most of the run HEALTHY
    - reach a max-x close to the target (catches runs that tracked
      partially then diverged to infinity)
    """
    endpoint = m.endpoint_error_m if not math.isnan(m.endpoint_error_m) else 5.0
    # Penalize runs that never got anywhere close to the target x.
    progress_gap = max(0.0, end_x_target - m.max_x_reached)
    return endpoint + 1.5 * progress_gap + 2.0 * (1.0 - m.frac_healthy) + 3.0 * m.frac_broken


# ---------------------------------------------------------------------------
# CLI: eval / sweep subcommands
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, default=None, help="Ground-truth JSON (defaults to <dataset>/ground_truth.json).")
    p.add_argument("--backend", default=_DEFAULT_DEPTH_BACKEND)
    p.add_argument("--extrinsics", type=Path, default=Path(_DEFAULT_EXTRINSICS))
    p.add_argument("--intrinsics", type=Path, default=Path(_DEFAULT_INTRINSICS))
    p.add_argument("--depth-scale", type=float, default=_DEFAULT_DEPTH_SCALE)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--rebuild-cache", action="store_true", help="Force depth re-inference.")


def _resolve_ground_truth(args: argparse.Namespace) -> GroundTruth:
    gt_path = args.ground_truth or (args.dataset / "ground_truth.json")
    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}", file=sys.stderr)
        sys.exit(2)
    return GroundTruth.from_json(gt_path)


def _load_obs(args: argparse.Namespace) -> Observations:
    return build_observations(
        args.dataset,
        backend=args.backend,
        depth_scale=args.depth_scale,
        intrinsics_path=args.intrinsics,
        extrinsics_path=args.extrinsics,
        max_frames=args.max_frames,
        force=args.rebuild_cache,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    gt = _resolve_ground_truth(args)
    obs = _load_obs(args)

    sm_cfg = ScanMatchConfig(
        residual_healthy=args.residual_healthy,
        residual_degraded=args.residual_degraded,
        min_correspondences=args.min_correspondences,
    )
    icp_cfg = ICPConfig(
        max_correspondence_dist=args.max_corr_dist,
        min_valid_bins=args.min_valid_bins,
    )
    scan_cfg = ScanConfig()
    metrics = evaluate_run(obs, scan_cfg=scan_cfg, icp_cfg=icp_cfg, sm_cfg=sm_cfg, ground_truth=gt)
    _print_metrics(metrics, label="eval")


def _cmd_sweep(args: argparse.Namespace) -> None:
    gt = _resolve_ground_truth(args)
    obs = _load_obs(args)

    residual_healthy = _parse_float_list(args.residual_healthy)
    residual_degraded = _parse_float_list(args.residual_degraded)
    max_corr = _parse_float_list(args.max_corr_dist)
    min_corr = _parse_int_list(args.min_correspondences)
    broken_thresh = _parse_int_list(args.degraded_to_broken)

    combos = list(itertools.product(residual_healthy, residual_degraded, max_corr, min_corr, broken_thresh))
    # Drop nonsense combos where healthy >= degraded.
    combos = [c for c in combos if c[0] < c[1]]

    target_x = gt.end_xy_m[0]
    base_scan = ScanConfig()
    results: list[tuple[tuple[float, float, float, int, int], EvalMetrics]] = []
    print(f"Running {len(combos)} combos over {obs.n_frames} frames...")
    t0 = time.monotonic()
    for i, (rh, rd, mc, mn, db) in enumerate(combos, 1):
        sm_cfg = ScanMatchConfig(
            residual_healthy=rh,
            residual_degraded=rd,
            min_correspondences=mn,
            degraded_to_broken_frames=db,
        )
        icp_cfg = ICPConfig(
            max_correspondence_dist=mc,
            min_valid_bins=mn,
        )
        m = evaluate_run(obs, scan_cfg=base_scan, icp_cfg=icp_cfg, sm_cfg=sm_cfg, ground_truth=gt)
        results.append(((rh, rd, mc, mn, db), m))
        if i % 10 == 0:
            dt = time.monotonic() - t0
            eta = dt / i * (len(combos) - i)
            print(f"  {i}/{len(combos)}  elapsed={dt:.0f}s  eta={eta:.0f}s")

    results.sort(key=lambda rm: _score(rm[1], target_x))
    top_n = min(args.top, len(results))
    print()
    print(f"Top {top_n} configs (lower score is better; target end_x = {target_x:+.2f} m):")
    header = f"{'rh':>6} {'rd':>6} {'mcd':>6} {'minc':>5} {'d2b':>4}   {'score':>6}  {'endpt':>6} {'maxx':>6} {'yaw':>6}  {'%hlt':>5} {'%brk':>5} {'%xy':>5}  {'r50':>5} {'r95':>5} {'inf':>5}"
    print(header)
    print("-" * len(header))
    for (rh, rd, mc, mn, db), m in results[:top_n]:
        print(
            f"{rh:>6.3f} {rd:>6.3f} {mc:>6.2f} {mn:>5d} {db:>4d}   "
            f"{_score(m, target_x):>6.3f}  {m.endpoint_error_m:>6.3f} {m.max_x_reached:>6.3f} {m.heading_error_deg:>+6.1f}  "
            f"{m.frac_healthy:>5.0%} {m.frac_broken:>5.0%} {m.frac_xy_valid:>5.0%}  "
            f"{m.residual_p50:>5.3f} {m.residual_p95:>5.3f} {m.residual_inf_rate:>5.0%}"
        )

    if args.save_json:
        out = []
        for (rh, rd, mc, mn, db), m in results:
            out.append({
                "config": {
                    "residual_healthy": rh,
                    "residual_degraded": rd,
                    "max_correspondence_dist": mc,
                    "min_correspondences": mn,
                    "degraded_to_broken_frames": db,
                },
                "score": _score(m, target_x),
                "metrics": m.as_row(),
            })
        args.save_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.save_json}")

    if results:
        (rh, rd, mc, mn, db), _m = results[0]
        print("\nSuggested defaults:")
        print(f"  ScanMatchConfig.residual_healthy = {rh}")
        print(f"  ScanMatchConfig.residual_degraded = {rd}")
        print(f"  ScanMatchConfig.min_correspondences = {mn}")
        print(f"  ScanMatchConfig.degraded_to_broken_frames = {db}")
        print(f"  ICPConfig.max_correspondence_dist = {mc}")
        print(f"  ICPConfig.min_valid_bins = {mn}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline scan-match evaluator / sweeper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Run one config and print metrics.")
    _add_common_args(p_eval)
    p_eval.add_argument("--residual-healthy", type=float, default=ScanMatchConfig.residual_healthy)
    p_eval.add_argument("--residual-degraded", type=float, default=ScanMatchConfig.residual_degraded)
    p_eval.add_argument("--min-correspondences", type=int, default=ScanMatchConfig.min_correspondences)
    p_eval.add_argument("--max-corr-dist", type=float, default=ICPConfig.max_correspondence_dist)
    p_eval.add_argument("--min-valid-bins", type=int, default=ICPConfig.min_valid_bins)
    p_eval.set_defaults(func=_cmd_eval)

    p_sweep = sub.add_parser("sweep", help="Grid-search configs and rank.")
    _add_common_args(p_sweep)
    p_sweep.add_argument("--residual-healthy", type=str, default="0.05,0.08,0.12,0.18")
    p_sweep.add_argument("--residual-degraded", type=str, default="0.15,0.25,0.35")
    p_sweep.add_argument("--max-corr-dist", type=str, default="0.5,1.0,1.5")
    p_sweep.add_argument("--min-correspondences", type=str, default="10,15,20")
    p_sweep.add_argument("--degraded-to-broken", type=str, default="10,20,40")
    p_sweep.add_argument("--top", type=int, default=15)
    p_sweep.add_argument("--save-json", type=Path, default=None)
    p_sweep.set_defaults(func=_cmd_sweep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
