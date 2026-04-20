from __future__ import annotations

from tankbot.desktop.autonomy.reactive.policy import PolicyDecision, WanderState
from tankbot.desktop.autonomy.reactive.pose import HealthState, PoseEstimate
from tankbot.desktop.autonomy.reactive.safety import SafetyTick
from tankbot.desktop.autonomy.vision import (
    REACTIVE_SAFETY_REVERSE_DURATION,
    REACTIVE_SAFETY_TURN_DURATION,
    _reactive_safety_override,
)


def _tick(state: WanderState, *, blocked: bool = True) -> SafetyTick:
    return SafetyTick(
        t_monotonic=1.0,
        pose=PoseEstimate(
            t_monotonic=1.0,
            yaw_rad=0.0,
            yaw_var=0.0,
            xy_m=None,
            xy_var=None,
            health=HealthState.HEALTHY,
            source="test",
        ),
        n_obstacle_points=100,
        stop_zone_blocked=blocked,
        stop_zone_count=3 if blocked else 0,
        near_field_veto=False,
        clearance_forward_m=0.2 if blocked else 1.5,
        clearance_left_m=1.0,
        clearance_right_m=0.8,
        ultrasonic_m=0.3 if blocked else 1.2,
        decision=PolicyDecision(
            left_duty=-1500 if state == WanderState.TURNING else -1200 if state == WanderState.REVERSING else 0,
            right_duty=1500 if state == WanderState.TURNING else -1200 if state == WanderState.REVERSING else 0,
            reason=f"reactive {state.value}",
            state=state,
        ),
        projection_ms=1.0,
        grid_ms=1.0,
        total_ms=2.0,
    )


def test_reactive_safety_override_passes_clear_forward() -> None:
    maneuver = _reactive_safety_override(1400, 1400, 0.25, _tick(WanderState.FORWARD, blocked=False))

    assert maneuver.left == 1400
    assert maneuver.right == 1400
    assert maneuver.duration_s == 0.25
    assert maneuver.remember is True
    assert maneuver.overridden is False


def test_reactive_safety_override_turns_blocked_forward() -> None:
    maneuver = _reactive_safety_override(1400, 1400, 0.40, _tick(WanderState.TURNING))

    assert maneuver.left == -1500
    assert maneuver.right == 1500
    assert maneuver.duration_s == REACTIVE_SAFETY_TURN_DURATION
    assert maneuver.remember is False
    assert maneuver.overridden is True


def test_reactive_safety_override_reverses_blocked_forward() -> None:
    maneuver = _reactive_safety_override(1400, 1400, 0.05, _tick(WanderState.REVERSING))

    assert maneuver.left == -1200
    assert maneuver.right == -1200
    assert maneuver.duration_s == REACTIVE_SAFETY_REVERSE_DURATION
    assert maneuver.remember is False
    assert maneuver.overridden is True


def test_reactive_safety_override_does_not_touch_turn_commands() -> None:
    maneuver = _reactive_safety_override(-1500, 1500, 0.20, _tick(WanderState.TURNING))

    assert maneuver.left == -1500
    assert maneuver.right == 1500
    assert maneuver.duration_s == 0.20
    assert maneuver.remember is True
    assert maneuver.overridden is False
