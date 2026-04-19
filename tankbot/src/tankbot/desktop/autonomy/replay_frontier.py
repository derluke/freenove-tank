"""Replay harness for the rolling frontier grid — Phase 3 §3.3.

Feeds a recorded dataset through the Phase 2 scan-match pose source and
into the new ``RollingFrontierGrid``, records health transitions, and
optionally renders each update as a PNG of the current window.

Depth inference is delegated to the shared ``Observations`` cache in
``scanmatch_eval`` so replays are fast once the cache is warm — no
depth-model reload, no re-projection.

Usage:
    tankbot-replay-frontier --dataset recordings/phase0c/straight_2m
    tankbot-replay-frontier --dataset <path> --render-dir logs/frontier_replay
    tankbot-replay-frontier --dataset-root recordings/phase0c --max-frames 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tankbot.desktop.autonomy.dataset import is_stream_dataset
from tankbot.desktop.autonomy.frontier import PlannerMode, PlannerSnapshot
from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate
from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)
from tankbot.desktop.autonomy.rolling_frontier import (
    RollingFrontierPlanner,
    RollingGridConfig,
)
from tankbot.desktop.autonomy.scanmatch_eval import (
    _DEFAULT_DEPTH_BACKEND,
    _DEFAULT_DEPTH_SCALE,
    _DEFAULT_EXTRINSICS,
    _DEFAULT_INTRINSICS,
    GroundTruth,
    Observations,
    build_observations,
)


@dataclass(frozen=True)
class HealthTransition:
    frame_idx: int
    t_monotonic: float
    prev: HealthState
    curr: HealthState


@dataclass
class ReplaySummary:
    n_frames: int
    health_counts: dict[HealthState, int]
    transitions: list[HealthTransition]
    final_pose: PoseEstimate
    final_command_mode: PlannerMode
    final_command_reason: str
    final_command_target_cell: tuple[int, int] | None
    final_selected_frontier_cell: tuple[int, int] | None
    final_anchor_xy: tuple[float, float]
    final_free_cells: int
    final_occupied_cells: int
    final_unknown_cells: int
    final_frontier_clusters: int
    total_writes: int
    recenters: int
    command_counts: dict[PlannerMode, int]
    degraded_continue_ticks: int
    degraded_hold_ticks: int
    final_snapshot: PlannerSnapshot


@dataclass(frozen=True)
class SweepRow:
    dataset: str
    n_frames: int
    final_health: str
    final_planner: str
    final_reason: str
    frac_healthy: float
    frac_degraded: float
    frac_broken: float
    frac_hold: float
    frac_turn: float
    frac_forward: float
    frac_frontier: float
    degraded_continue_ticks: int
    degraded_hold_ticks: int
    recenters: int
    frontier_clusters: int
    endpoint_error_m: float | None
    heading_error_deg: float | None

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_frames": self.n_frames,
            "final_health": self.final_health,
            "final_planner": self.final_planner,
            "final_reason": self.final_reason,
            "frac_healthy": self.frac_healthy,
            "frac_degraded": self.frac_degraded,
            "frac_broken": self.frac_broken,
            "frac_hold": self.frac_hold,
            "frac_turn": self.frac_turn,
            "frac_forward": self.frac_forward,
            "frac_frontier": self.frac_frontier,
            "degraded_continue_ticks": self.degraded_continue_ticks,
            "degraded_hold_ticks": self.degraded_hold_ticks,
            "recenters": self.recenters,
            "frontier_clusters": self.frontier_clusters,
            "endpoint_error_m": self.endpoint_error_m,
            "heading_error_deg": self.heading_error_deg,
        }


def _colorize_state(state: np.ndarray) -> np.ndarray:
    """Map ``state`` (0/1/2) to a BGR image. 0=dark gray, 1=white, 2=black."""
    h, w = state.shape
    img = np.full((h, w, 3), 64, dtype=np.uint8)
    img[state == 1] = (230, 230, 230)
    img[state == 2] = (0, 0, 0)
    return img


def _draw_frontier(img: np.ndarray, planner: RollingFrontierPlanner) -> None:
    for cluster in planner.grid.frontier_clusters():
        for row, col in cluster:
            img[row, col] = (0, 0, 220)


def _render(
    planner: RollingFrontierPlanner,
    pose: PoseEstimate,
    upscale: int = 5,
) -> np.ndarray:
    """Render a single-frame snapshot panel.

    Grid is upscaled with nearest-neighbor so cell edges stay crisp. The
    robot marker (green dot) is drawn after upscaling so it stays a fixed
    screen size regardless of grid resolution.
    """
    import cv2

    grid = planner.grid
    state = grid.state_grid()
    img = _colorize_state(state)
    _draw_frontier(img, planner)

    h, w = img.shape[:2]
    big = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)

    if pose.xy_m is not None:
        cell = grid.world_to_cell(*pose.xy_m)
        if cell is not None:
            row, col = cell
            cx = col * upscale + upscale // 2
            cy = row * upscale + upscale // 2
            cv2.circle(big, (cx, cy), 4, (0, 200, 0), -1)
            # Heading indicator.
            hx = int(cx + math.cos(pose.yaw_rad) * 12)
            hy = int(cy + math.sin(pose.yaw_rad) * 12)
            cv2.line(big, (cx, cy), (hx, hy), (0, 200, 0), 2)

    # Anchor cross at window center.
    ax = (grid.n_cells // 2) * upscale + upscale // 2
    ay = (grid.n_cells // 2) * upscale + upscale // 2
    cv2.drawMarker(big, (ax, ay), (220, 120, 0), cv2.MARKER_CROSS, 12, 1)

    # Text overlay with anchor + health.
    anchor = grid.anchor_xy_m
    lines = [
        f"anchor=({anchor[0]:+.2f}, {anchor[1]:+.2f}) m",
        f"health={pose.health.value}",
        f"planner={planner.snapshot().planner_mode}",
        f"pose=({pose.xy_m[0]:+.3f}, {pose.xy_m[1]:+.3f}) m" if pose.xy_m else "pose=None",
        f"yaw={math.degrees(pose.yaw_rad):+.1f} deg",
    ]
    y = 18
    for line in lines:
        cv2.putText(big, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16

    return big


def replay(
    obs: Observations,
    *,
    grid_config: RollingGridConfig | None = None,
    scan_config: ScanConfig | None = None,
    icp_config: ICPConfig | None = None,
    sm_config: ScanMatchConfig | None = None,
    render_dir: Path | None = None,
    render_every: int = 10,
    upscale: int = 5,
    progress_every: int = 100,
) -> ReplaySummary:
    """Run the pose-source → rolling grid pipeline over a cached dataset."""
    pose_source = ScanMatchPoseSource(
        scan_config=scan_config or ScanConfig(),
        icp_config=icp_config or ICPConfig(),
        config=sm_config or ScanMatchConfig(),
    )
    planner = RollingFrontierPlanner(grid_config)

    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    health_counts: dict[HealthState, int] = {
        HealthState.HEALTHY: 0,
        HealthState.DEGRADED: 0,
        HealthState.BROKEN: 0,
    }
    transitions: list[HealthTransition] = []
    command_counts: dict[PlannerMode, int] = {mode: 0 for mode in PlannerMode}
    degraded_continue_ticks = 0
    degraded_hold_ticks = 0
    prev_health: HealthState | None = None
    prev_anchor: tuple[float, float] | None = None
    recenters = 0
    final_command = None

    for i in range(obs.n_frames):
        points = obs.frame_points(i)
        gyro_yaw = float(obs.gyro_yaw[i])
        gyro_arg = None if math.isnan(gyro_yaw) else gyro_yaw
        pose_source.update(
            points,
            gyro_yaw=gyro_arg,
            motors_active=bool(obs.motors_active[i]),
            t_monotonic=float(obs.t_monotonic[i]),
        )
        pose = pose_source.latest()

        planner.update_from_frame(pose, points)
        command = planner.command_for_pose(pose)
        final_command = command
        command_counts[command.mode] = command_counts.get(command.mode, 0) + 1
        if pose.health == HealthState.DEGRADED:
            if command.reason == "pose_degraded_continue":
                degraded_continue_ticks += 1
            elif command.mode == PlannerMode.HOLD:
                degraded_hold_ticks += 1

        health_counts[pose.health] = health_counts.get(pose.health, 0) + 1
        if prev_health is None or pose.health != prev_health:
            if prev_health is not None:
                transitions.append(
                    HealthTransition(
                        frame_idx=int(obs.frame_idx[i]),
                        t_monotonic=float(obs.t_monotonic[i]),
                        prev=prev_health,
                        curr=pose.health,
                    )
            )
            prev_health = pose.health

        anchor = planner.grid.anchor_xy_m
        if prev_anchor is not None and anchor != prev_anchor:
            recenters += 1
        prev_anchor = anchor

        if render_dir is not None and i % render_every == 0:
            import cv2

            panel = _render(planner, pose, upscale=upscale)
            cv2.imwrite(str(render_dir / f"frontier_{i:06d}.png"), panel)

        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  frame {i + 1}/{obs.n_frames}  health={pose.health.value:<8}  "
                f"planner={command.mode.value:<17}  "
                f"anchor=({anchor[0]:+.2f},{anchor[1]:+.2f})  recenters={recenters}",
            )

    snap = planner.grid.snapshot()
    planner_snapshot = planner.snapshot()
    final_pose = pose_source.latest()
    assert final_command is not None
    return ReplaySummary(
        n_frames=obs.n_frames,
        health_counts=health_counts,
        transitions=transitions,
        final_pose=final_pose,
        final_command_mode=final_command.mode,
        final_command_reason=final_command.reason,
        final_command_target_cell=final_command.target_cell,
        final_selected_frontier_cell=planner.selected_frontier_cell,
        final_anchor_xy=snap.anchor_xy_m,
        final_free_cells=snap.free_cells,
        final_occupied_cells=snap.occupied_cells,
        final_unknown_cells=snap.unknown_cells,
        final_frontier_clusters=len(planner.grid.frontier_clusters()),
        total_writes=snap.writes_since_reset,
        recenters=recenters,
        command_counts=command_counts,
        degraded_continue_ticks=degraded_continue_ticks,
        degraded_hold_ticks=degraded_hold_ticks,
        final_snapshot=planner_snapshot,
    )


def _print_summary(s: ReplaySummary) -> None:
    total = max(s.n_frames, 1)
    print()
    print("=== Rolling frontier replay summary ===")
    print(f"  frames:            {s.n_frames}")
    print(f"  final health:      {s.final_pose.health.value}")
    print(f"  final planner:     {s.final_command_mode.value} ({s.final_command_reason})")
    print(
        "  health mix:        "
        f"healthy={s.health_counts[HealthState.HEALTHY] / total:.0%}  "
        f"degraded={s.health_counts[HealthState.DEGRADED] / total:.0%}  "
        f"broken={s.health_counts[HealthState.BROKEN] / total:.0%}",
    )
    print(
        "  planner mix:       "
        f"hold={s.command_counts[PlannerMode.HOLD] / total:.0%}  "
        f"turn={s.command_counts[PlannerMode.TURN] / total:.0%}  "
        f"forward={s.command_counts[PlannerMode.FORWARD] / total:.0%}  "
        f"frontier={s.command_counts[PlannerMode.APPROACH_FRONTIER] / total:.0%}",
    )
    if s.health_counts[HealthState.DEGRADED] > 0:
        degraded_total = s.health_counts[HealthState.DEGRADED]
        print(
            "  degraded policy:   "
            f"continue={s.degraded_continue_ticks}/{degraded_total}  "
            f"hold={s.degraded_hold_ticks}/{degraded_total}",
        )
    if s.final_pose.xy_m is not None:
        fx, fy = s.final_pose.xy_m
        print(f"  final pose:        ({fx:+.3f}, {fy:+.3f}) m  yaw={math.degrees(s.final_pose.yaw_rad):+.2f} deg")
    else:
        print(f"  final pose:        None  yaw={math.degrees(s.final_pose.yaw_rad):+.2f} deg")
    ax, ay = s.final_anchor_xy
    print(f"  final anchor:      ({ax:+.2f}, {ay:+.2f}) m")
    print(f"  window recenters:  {s.recenters}")
    print(f"  grid writes:       {s.total_writes}")
    print(
        "  final cells:       "
        f"free={s.final_free_cells}  occ={s.final_occupied_cells}  unknown={s.final_unknown_cells}",
    )
    print(f"  frontier clusters: {s.final_frontier_clusters}")
    if s.final_snapshot.selected_frontier is not None:
        tx, ty = s.final_snapshot.selected_frontier
        print(
            "  selected frontier: "
            f"({tx:+.2f}, {ty:+.2f}) m  cell={s.final_selected_frontier_cell}"
        )
        print(f"  command target:    {s.final_command_target_cell}")
    else:
        print("  selected frontier: none")

    if s.transitions:
        print()
        print(f"  health transitions ({len(s.transitions)}):")
        for t in s.transitions:
            print(f"    frame {t.frame_idx:>5d}  t={t.t_monotonic:+.2f}s  {t.prev.value} -> {t.curr.value}")
    else:
        print("  health transitions: none")


def _discover_datasets(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        msg = f"Dataset root not found: {dataset_root}"
        raise FileNotFoundError(msg)
    return sorted(
        p for p in dataset_root.iterdir()
        if p.is_dir() and is_stream_dataset(p)
    )


def _compute_optional_ground_truth_metrics(dataset: Path, summary: ReplaySummary) -> tuple[float | None, float | None]:
    gt_path = dataset / "ground_truth.json"
    if not gt_path.exists():
        return None, None
    gt = GroundTruth.from_json(gt_path)
    xy = summary.final_pose.xy_m if summary.final_pose.xy_m is not None else None
    if xy is None:
        return None, None
    endpoint_error = math.hypot(xy[0] - gt.end_xy_m[0], xy[1] - gt.end_xy_m[1])
    heading_error = math.degrees(summary.final_pose.yaw_rad) - gt.end_yaw_deg
    return endpoint_error, heading_error


def _row_for_summary(dataset: Path, summary: ReplaySummary) -> SweepRow:
    total = max(summary.n_frames, 1)
    endpoint_error, heading_error = _compute_optional_ground_truth_metrics(dataset, summary)
    return SweepRow(
        dataset=dataset.name,
        n_frames=summary.n_frames,
        final_health=summary.final_pose.health.value,
        final_planner=summary.final_command_mode.value,
        final_reason=summary.final_command_reason,
        frac_healthy=summary.health_counts[HealthState.HEALTHY] / total,
        frac_degraded=summary.health_counts[HealthState.DEGRADED] / total,
        frac_broken=summary.health_counts[HealthState.BROKEN] / total,
        frac_hold=summary.command_counts[PlannerMode.HOLD] / total,
        frac_turn=summary.command_counts[PlannerMode.TURN] / total,
        frac_forward=summary.command_counts[PlannerMode.FORWARD] / total,
        frac_frontier=summary.command_counts[PlannerMode.APPROACH_FRONTIER] / total,
        degraded_continue_ticks=summary.degraded_continue_ticks,
        degraded_hold_ticks=summary.degraded_hold_ticks,
        recenters=summary.recenters,
        frontier_clusters=summary.final_frontier_clusters,
        endpoint_error_m=endpoint_error,
        heading_error_deg=heading_error,
    )


def sweep_datasets(
    datasets: list[Path],
    *,
    backend: str,
    depth_scale: float,
    intrinsics_path: Path,
    extrinsics_path: Path,
    max_frames: int | None,
    rebuild_cache: bool,
    grid_config: RollingGridConfig,
    render_root: Path | None = None,
    render_every: int = 10,
    upscale: int = 5,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for dataset in datasets:
        render_dir = None if render_root is None else (render_root / dataset.name)
        obs = build_observations(
            dataset,
            backend=backend,
            depth_scale=depth_scale,
            intrinsics_path=intrinsics_path,
            extrinsics_path=extrinsics_path,
            max_frames=max_frames,
            force=rebuild_cache,
            verbose=True,
        )
        summary = replay(
            obs,
            grid_config=grid_config,
            render_dir=render_dir,
            render_every=render_every,
            upscale=upscale,
        )
        rows.append(_row_for_summary(dataset, summary))
    return rows


def _print_sweep(rows: list[SweepRow]) -> None:
    print()
    print("=== Rolling frontier sweep summary ===")
    if not rows:
        print("  no datasets")
        return
    header = (
        f"{'dataset':<22} {'frm':>5} {'hlt':>5} {'deg':>5} {'brk':>5} "
        f"{'hold':>5} {'turn':>5} {'frt':>5} {'d+':>4} {'dH':>4} "
        f"{'rec':>3} {'fc':>3} {'endpt':>6} {'yaw':>6}  {'final':<17} {'reason'}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        endpoint = f"{row.endpoint_error_m:>6.3f}" if row.endpoint_error_m is not None else "   n/a"
        yaw = f"{row.heading_error_deg:>+6.1f}" if row.heading_error_deg is not None else "   n/a"
        print(
            f"{row.dataset:<22} {row.n_frames:>5d} {row.frac_healthy:>5.0%} {row.frac_degraded:>5.0%} {row.frac_broken:>5.0%} "
            f"{row.frac_hold:>5.0%} {row.frac_turn:>5.0%} {row.frac_frontier:>5.0%} "
            f"{row.degraded_continue_ticks:>4d} {row.degraded_hold_ticks:>4d} "
            f"{row.recenters:>3d} {row.frontier_clusters:>3d} {endpoint} {yaw}  "
            f"{row.final_planner:<17} {row.final_reason}"
        )


def _save_sweep_json(path: Path, rows: list[SweepRow]) -> None:
    path.write_text(
        json.dumps([row.as_row() for row in rows], indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset into the rolling frontier grid.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=Path)
    dataset_group.add_argument("--dataset-root", type=Path, help="Sweep all dataset dirs beneath this root.")
    parser.add_argument("--backend", default=_DEFAULT_DEPTH_BACKEND)
    parser.add_argument("--extrinsics", type=Path, default=Path(_DEFAULT_EXTRINSICS))
    parser.add_argument("--intrinsics", type=Path, default=Path(_DEFAULT_INTRINSICS))
    parser.add_argument("--depth-scale", type=float, default=_DEFAULT_DEPTH_SCALE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--window-size-m", type=float, default=RollingGridConfig.window_size_m)
    parser.add_argument("--resolution-m", type=float, default=RollingGridConfig.resolution_m)
    parser.add_argument("--recenter-margin-m", type=float, default=RollingGridConfig.recenter_margin_m)
    parser.add_argument("--max-obs-range-m", type=float, default=RollingGridConfig.max_observation_range_m)

    parser.add_argument("--render-dir", type=Path, default=None, help="If set, write PNG panels here.")
    parser.add_argument("--render-every", type=int, default=10, help="Render every Nth frame.")
    parser.add_argument("--upscale", type=int, default=5, help="Nearest-neighbor upscale factor for panels.")
    parser.add_argument("--save-json", type=Path, default=None, help="If set, write sweep rows to JSON.")
    args = parser.parse_args(argv)

    grid_cfg = RollingGridConfig(
        window_size_m=args.window_size_m,
        resolution_m=args.resolution_m,
        recenter_margin_m=args.recenter_margin_m,
        max_observation_range_m=args.max_obs_range_m,
    )

    if args.dataset_root is not None:
        datasets = _discover_datasets(args.dataset_root)
        rows = sweep_datasets(
            datasets,
            backend=args.backend,
            depth_scale=args.depth_scale,
            intrinsics_path=args.intrinsics,
            extrinsics_path=args.extrinsics,
            max_frames=args.max_frames,
            rebuild_cache=args.rebuild_cache,
            grid_config=grid_cfg,
            render_root=args.render_dir,
            render_every=args.render_every,
            upscale=args.upscale,
        )
        _print_sweep(rows)
        if args.save_json is not None:
            _save_sweep_json(args.save_json, rows)
            print(f"\nFull results written to {args.save_json}")
        return

    assert args.dataset is not None
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    obs = build_observations(
        args.dataset,
        backend=args.backend,
        depth_scale=args.depth_scale,
        intrinsics_path=args.intrinsics,
        extrinsics_path=args.extrinsics,
        max_frames=args.max_frames,
        force=args.rebuild_cache,
    )
    summary = replay(
        obs,
        grid_config=grid_cfg,
        render_dir=args.render_dir,
        render_every=args.render_every,
        upscale=args.upscale,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
