"""Goal and behavior scaffolding above the low-level autonomy FSM.

This module does not decide wheel commands directly. It describes the
high-level intent of the robot and maps that intent onto reusable
behaviors, which the low-level controller can then execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GoalKind(str, Enum):
    EXPLORE_ROOM = "explore_room"
    FIND_BOUNDARY = "find_boundary"
    GO_TO_POSE = "go_to_pose"
    FIND_OBJECT = "find_object"


class BehaviorKind(str, Enum):
    BOOTSTRAP_MAP = "bootstrap_map"
    INITIAL_SCAN = "initial_scan"
    EXPLORE_SPACE = "explore_space"
    RECOVER_TRACKING = "recover_tracking"
    SEEK_BOUNDARY = "seek_boundary"
    NAVIGATE_TO_POSE = "navigate_to_pose"
    SEARCH_FOR_OBJECT = "search_for_object"


@dataclass(frozen=True)
class Goal:
    kind: GoalKind
    params: dict[str, Any] | None = None
    description: str = ""


@dataclass(frozen=True)
class PlanningSnapshot:
    goal: Goal
    behavior: BehaviorKind
    phase: str


class AutonomyPlanner:
    """Maps a high-level goal onto the active behavior for the controller."""

    def __init__(self, goal: Goal | None = None) -> None:
        self._goal = goal or Goal(
            kind=GoalKind.EXPLORE_ROOM,
            description="Explore the room while building and maintaining a map.",
        )

    @property
    def goal(self) -> Goal:
        return self._goal

    def set_goal(self, goal: Goal) -> None:
        self._goal = goal

    def snapshot(self, *, phase: str) -> PlanningSnapshot:
        behavior = self._behavior_for_phase(self._goal.kind, phase)
        return PlanningSnapshot(
            goal=self._goal,
            behavior=behavior,
            phase=phase,
        )

    def _behavior_for_phase(self, goal: GoalKind, phase: str) -> BehaviorKind:
        if phase == "bootstrap":
            return BehaviorKind.BOOTSTRAP_MAP
        if phase == "initial_scan":
            return BehaviorKind.INITIAL_SCAN
        if phase == "recovering":
            return BehaviorKind.RECOVER_TRACKING

        if goal == GoalKind.FIND_BOUNDARY:
            return BehaviorKind.SEEK_BOUNDARY
        if goal == GoalKind.GO_TO_POSE:
            return BehaviorKind.NAVIGATE_TO_POSE
        if goal == GoalKind.FIND_OBJECT:
            return BehaviorKind.SEARCH_FOR_OBJECT
        return BehaviorKind.EXPLORE_SPACE
