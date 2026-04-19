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
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate
from tankbot.desktop.autonomy.reactive.scan import ScanConfig
from tankbot.desktop.autonomy.reactive.scan_match import ICPConfig
from tankbot.desktop.autonomy.reactive.scan_match_pose import (
    ScanMatchConfig,
    ScanMatchPoseSource,
)
from tankbot.desktop.autonomy.rolling_frontier import (
    RollingFrontierGrid,
    RollingGridConfig,
)
from tankbot.desktop.autonomy.scanmatch_eval import (
    _DEFAULT_DEPTH_BACKEND,
    _DEFAULT_DEPTH_SCALE,
    _DEFAULT_EXTRINSICS,
    _DEFAULT_INTRINSICS,
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
    final_anchor_xy: tuple[float, float]
    final_free_cells: int
    final_occupied_cells: int
    final_unknown_cells: int
    final_frontier_clusters: int
    total_writes: int
    recenters: int


def _colorize_state(state: np.ndarray) -> np.ndarray:
    """Map ``state`` (0/1/2) to a BGR image. 0=dark gray, 1=white, 2=black."""
    h, w = state.shape
    img = np.full((h, w, 3), 64, dtype=np.uint8)
    img[state == 1] = (230, 230, 230)
    img[state == 2] = (0, 0, 0)
    return img


def _draw_frontier(img: np.ndarray, grid: RollingFrontierGrid) -> None:
    for cluster in grid.frontier_clusters():
        for row, col in cluster:
            img[row, col] = (0, 0, 220)


def _render(
    grid: RollingFrontierGrid,
    pose: PoseEstimate,
    upscale: int = 5,
) -> np.ndarray:
    """Render a single-frame snapshot panel.

    Grid is upscaled with nearest-neighbor so cell edges stay crisp. The
    robot marker (green dot) is drawn after upscaling so it stays a fixed
    screen size regardless of grid resolution.
    """
    import cv2

    state = grid.state_grid()
    img = _colorize_state(state)
    _draw_frontier(img, grid)

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
    grid = RollingFrontierGrid(grid_config)

    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    health_counts: dict[HealthState, int] = {
        HealthState.HEALTHY: 0,
        HealthState.DEGRADED: 0,
        HealthState.BROKEN: 0,
    }
    transitions: list[HealthTransition] = []
    prev_health: HealthState | None = None
    prev_anchor: tuple[float, float] | None = None
    recenters = 0

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

        grid.update_from_frame(pose, points)

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

        anchor = grid.anchor_xy_m
        if prev_anchor is not None and anchor != prev_anchor:
            recenters += 1
        prev_anchor = anchor

        if render_dir is not None and i % render_every == 0:
            import cv2

            panel = _render(grid, pose, upscale=upscale)
            cv2.imwrite(str(render_dir / f"frontier_{i:06d}.png"), panel)

        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  frame {i + 1}/{obs.n_frames}  health={pose.health.value:<8}  "
                f"anchor=({anchor[0]:+.2f},{anchor[1]:+.2f})  recenters={recenters}",
            )

    snap = grid.snapshot()
    final_pose = pose_source.latest()
    return ReplaySummary(
        n_frames=obs.n_frames,
        health_counts=health_counts,
        transitions=transitions,
        final_pose=final_pose,
        final_anchor_xy=snap.anchor_xy_m,
        final_free_cells=snap.free_cells,
        final_occupied_cells=snap.occupied_cells,
        final_unknown_cells=snap.unknown_cells,
        final_frontier_clusters=len(grid.frontier_clusters()),
        total_writes=snap.writes_since_reset,
        recenters=recenters,
    )


def _print_summary(s: ReplaySummary) -> None:
    total = max(s.n_frames, 1)
    print()
    print("=== Rolling frontier replay summary ===")
    print(f"  frames:            {s.n_frames}")
    print(f"  final health:      {s.final_pose.health.value}")
    print(
        "  health mix:        "
        f"healthy={s.health_counts[HealthState.HEALTHY] / total:.0%}  "
        f"degraded={s.health_counts[HealthState.DEGRADED] / total:.0%}  "
        f"broken={s.health_counts[HealthState.BROKEN] / total:.0%}",
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

    if s.transitions:
        print()
        print(f"  health transitions ({len(s.transitions)}):")
        for t in s.transitions:
            print(f"    frame {t.frame_idx:>5d}  t={t.t_monotonic:+.2f}s  {t.prev.value} -> {t.curr.value}")
    else:
        print("  health transitions: none")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset into the rolling frontier grid.")
    parser.add_argument("--dataset", type=Path, required=True)
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
    args = parser.parse_args(argv)

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

    grid_cfg = RollingGridConfig(
        window_size_m=args.window_size_m,
        resolution_m=args.resolution_m,
        recenter_margin_m=args.recenter_margin_m,
        max_observation_range_m=args.max_obs_range_m,
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
