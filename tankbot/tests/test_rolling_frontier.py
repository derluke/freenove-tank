from __future__ import annotations

import numpy as np

from tankbot.desktop.autonomy.frontier import PlannerMode
from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate
from tankbot.desktop.autonomy.rolling_frontier import RollingFrontierGrid, RollingFrontierPlanner


def _pose(
    *,
    x: float,
    y: float,
    t_monotonic: float = 0.0,
    yaw_rad: float = 0.0,
    health: HealthState = HealthState.HEALTHY,
) -> PoseEstimate:
    return PoseEstimate(
        t_monotonic=t_monotonic,
        yaw_rad=yaw_rad,
        yaw_var=0.0,
        xy_m=(x, y),
        xy_var=0.0,
        health=health,
        source="test",
    )


def test_rolling_grid_ignores_non_healthy_pose() -> None:
    grid = RollingFrontierGrid()
    points = np.array([[1.0, 0.0, 0.2]], dtype=np.float32)

    grid.update_from_frame(_pose(x=0.0, y=0.0, health=HealthState.BROKEN), points)

    snap = grid.snapshot()
    assert snap.free_cells == 0
    assert snap.occupied_cells == 0
    assert snap.writes_since_reset == 0


def test_rolling_grid_recenters_with_robot_motion() -> None:
    grid = RollingFrontierGrid()
    grid.update_from_frame(_pose(x=3.5, y=0.0), np.array([[1.0, 0.0, 0.2]], dtype=np.float32))

    anchor_x, anchor_y = grid.anchor_xy_m
    assert anchor_x > 0.0
    assert anchor_y == 0.0


def test_planner_holds_when_translation_unavailable() -> None:
    planner = RollingFrontierPlanner()
    pose = PoseEstimate(
        t_monotonic=0.0,
        yaw_rad=0.0,
        yaw_var=0.0,
        xy_m=None,
        xy_var=None,
        health=HealthState.HEALTHY,
        source="test",
    )

    cmd = planner.command_for_pose(pose)

    assert cmd.mode == PlannerMode.HOLD
    assert cmd.reason == "translation_unavailable"


def test_planner_selects_forward_frontier() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid

    # Known free patch with unknown cells directly in front of it creates a
    # frontier cluster centered ahead of the robot.
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001
    grid._occ_hits.fill(0)  # noqa: SLF001

    cmd = planner.command_for_pose(_pose(x=0.0, y=0.0))

    assert cmd.mode in {PlannerMode.FORWARD, PlannerMode.APPROACH_FRONTIER, PlannerMode.TURN}
    assert cmd.target_cell is not None
    assert cmd.reason in {"approach_frontier", "align_frontier", "frontier_reached_reorient"}


def test_planner_snapshot_reports_selected_frontier() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001

    planner.command_for_pose(_pose(x=0.0, y=0.0))
    snap = planner.snapshot()

    assert snap.frontier_count >= 1
    assert snap.selected_frontier is not None


def test_planner_turns_when_only_frontiers_are_too_close() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 60:63] = 3  # noqa: SLF001

    cmd = planner.command_for_pose(_pose(x=0.0, y=0.0))

    assert cmd.mode == PlannerMode.TURN
    assert cmd.reason == "frontier_too_close"


def test_planner_turns_during_recent_turn_in_place_motion() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001

    planner.command_for_pose(_pose(x=0.0, y=0.0, t_monotonic=0.0, yaw_rad=0.0))
    cmd = planner.command_for_pose(_pose(x=0.03, y=0.0, t_monotonic=0.5, yaw_rad=0.20))

    assert cmd.mode == PlannerMode.TURN
    assert cmd.reason == "recent_turn_motion"


def test_planner_continues_inflight_target_during_short_degraded_handoff() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001

    healthy_cmd = planner.command_for_pose(_pose(x=0.0, y=0.0))
    degraded_pose = _pose(x=0.05, y=0.0, t_monotonic=0.2, health=HealthState.DEGRADED)

    assert healthy_cmd.target_cell is not None
    cmd = planner.command_for_pose(degraded_pose)

    assert cmd.mode == healthy_cmd.mode
    assert cmd.reason == "pose_degraded_continue"
    assert cmd.target_cell == healthy_cmd.target_cell
    assert planner.selected_frontier_cell is not None
    assert planner.snapshot().selected_frontier is not None


def test_planner_degraded_handoff_times_out_to_hold() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001

    planner.command_for_pose(_pose(x=0.0, y=0.0))
    planner.command_for_pose(_pose(x=0.0, y=0.0, t_monotonic=0.0, health=HealthState.DEGRADED))
    cmd = planner.command_for_pose(_pose(x=0.05, y=0.0, t_monotonic=1.0, health=HealthState.DEGRADED))

    assert cmd.mode == PlannerMode.HOLD
    assert cmd.reason == "pose_degraded_timeout"


def test_planner_requires_stable_healthy_window_after_degraded() -> None:
    planner = RollingFrontierPlanner()
    grid = planner.grid
    grid._free_hits[58:61, 68:71] = 3  # noqa: SLF001

    first_cmd = planner.command_for_pose(_pose(x=0.0, y=0.0))
    assert first_cmd.target_cell is not None

    degraded_cmd = planner.command_for_pose(_pose(x=0.0, y=0.0, t_monotonic=0.1, health=HealthState.DEGRADED))
    assert degraded_cmd.mode in {PlannerMode.HOLD, PlannerMode.TURN, PlannerMode.APPROACH_FRONTIER}

    for i in range(1, 4):
        cmd = planner.command_for_pose(_pose(x=0.0, y=0.0, t_monotonic=0.1 + i * 0.1))
        assert cmd.mode == PlannerMode.HOLD
        assert cmd.reason == "pose_recovering"

    recovered_cmd = planner.command_for_pose(_pose(x=0.0, y=0.0, t_monotonic=0.5))
    assert recovered_cmd.mode in {PlannerMode.FORWARD, PlannerMode.APPROACH_FRONTIER, PlannerMode.TURN}
    assert recovered_cmd.reason in {"approach_frontier", "align_frontier", "frontier_reached_reorient"}
