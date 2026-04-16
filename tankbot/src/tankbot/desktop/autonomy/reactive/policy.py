"""Reactive wander policy — stateless forward/turn decision.

Per the Phase 1 plan: "simple wander (forward until blocked, turn,
repeat, small random bias). Not optimal coverage, but reliable."

The policy is *stateless* in the sense that it does not carry hidden
state between ticks beyond the explicit `WanderState` enum (which the
caller persists so it survives across frames). Every decision is a
pure function of the current grid clearance + the caller-provided
state.

Motor output is expressed as a `PolicyDecision` with signed left/right
duty values. The reactive safety layer maps these to the actual motor
command; the policy itself never touches the robot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class WanderState(Enum):
    """High-level behavioral state for the wander policy."""

    FORWARD = "forward"
    TURNING = "turning"
    REVERSING = "reversing"
    STOPPED = "stopped"


@dataclass(frozen=True)
class PolicyDecision:
    """Motor intent produced by the wander policy.

    `left_duty` / `right_duty` are signed values in the motor driver's
    native range (roughly -4095 to +4095). Positive = forward.

    `reason` is a short human-readable string for replay visualization
    and logging — not parsed by downstream code.
    """

    left_duty: int
    right_duty: int
    reason: str
    state: WanderState


# Phase 1 speed cap: ~17 cm/s forward. Turn duty must be higher than
# forward because skid-steer in-place rotation fights tread scrub
# friction. MIN_DUTY=1000 is the deadband; 1500 is the floor for
# reliable turning on carpet, 1800 is the escalation ceiling.
DEFAULT_CRUISE_DUTY: int = 1200
DEFAULT_TURN_DUTY: int = 1500
TURN_ESCALATE_DUTY: int = 1800
TURN_ESCALATE_AFTER: int = 45  # frames of continuous turning before escalation (~2 s at 23 Hz)
REVERSE_AFTER: int = 90  # frames of continuous turning before reversing (~4 s at 23 Hz)
REVERSE_DUTY: int = 1200
REVERSE_FRAMES: int = 20  # reverse for ~0.9 s then try a new turn direction


class ReactiveWanderPolicy:
    """Wander: forward until blocked, turn to clear, repeat.

    Turn direction is chosen once when entering TURNING and committed to
    until the robot resumes forward motion.  If turning stalls (stop zone
    stays blocked for TURN_ESCALATE_AFTER frames), duty ramps up to
    TURN_ESCALATE_DUTY to overcome tread friction.

    If still stuck after REVERSE_AFTER frames of turning, the robot backs
    up for REVERSE_FRAMES then picks a fresh turn direction. This handles
    corners and tight spots where spinning in place can't find a way out.
    """

    def __init__(
        self,
        *,
        cruise_duty: int = DEFAULT_CRUISE_DUTY,
        turn_duty: int = DEFAULT_TURN_DUTY,
        turn_escalate_duty: int = TURN_ESCALATE_DUTY,
        turn_escalate_after: int = TURN_ESCALATE_AFTER,
        reverse_after: int = REVERSE_AFTER,
        reverse_duty: int = REVERSE_DUTY,
        reverse_frames: int = REVERSE_FRAMES,
        clearance_threshold_m: float = 0.50,
        random_bias: float = 0.15,
    ) -> None:
        self._cruise_duty = cruise_duty
        self._turn_duty = turn_duty
        self._turn_escalate_duty = turn_escalate_duty
        self._turn_escalate_after = turn_escalate_after
        self._reverse_after = reverse_after
        self._reverse_duty = reverse_duty
        self._reverse_frames = reverse_frames
        self._clearance_threshold_m = clearance_threshold_m
        self._random_bias = random_bias
        # Committed turn direction: +1 = left, -1 = right, 0 = none.
        self._committed_turn: int = 0
        # Consecutive frames spent turning (for escalation).
        self._turn_frames: int = 0
        # Frames spent reversing (counts down to 0).
        self._reverse_remaining: int = 0

    def decide(
        self,
        *,
        stop_zone_blocked: bool,
        clearance_left_m: float,
        clearance_right_m: float,
        clearance_forward_m: float,
        state: WanderState,
    ) -> PolicyDecision:
        """Decision function. Returns a `PolicyDecision`."""

        # If we're mid-reverse, keep backing up until the counter expires.
        if self._reverse_remaining > 0:
            self._reverse_remaining -= 1
            remaining = self._reverse_remaining
            if remaining == 0:
                # Done reversing — force a fresh turn direction next tick.
                self._committed_turn = 0
                self._turn_frames = 0
            return PolicyDecision(
                left_duty=-self._reverse_duty,
                right_duty=-self._reverse_duty,
                reason=f"reversing ({remaining} frames left)",
                state=WanderState.REVERSING,
            )

        # If the stop zone is blocked, we must turn (or stay turning).
        if stop_zone_blocked:
            return self._turn(clearance_left_m, clearance_right_m)

        # If we were turning, resume forward only when there is enough
        # forward clearance. This prevents oscillating at a threshold
        # boundary.
        if state == WanderState.TURNING and clearance_forward_m < self._clearance_threshold_m:
            return self._turn(clearance_left_m, clearance_right_m)

        # Resuming forward -- release turn state.
        self._committed_turn = 0
        self._turn_frames = 0
        return PolicyDecision(
            left_duty=self._cruise_duty,
            right_duty=self._cruise_duty,
            reason=f"forward (clearance {clearance_forward_m:.2f} m)",
            state=WanderState.FORWARD,
        )

    def _turn(self, clearance_left_m: float, clearance_right_m: float) -> PolicyDecision:
        """Pick (or continue) a turn direction, escalating duty if stuck.

        After `reverse_after` frames of continuous turning, trigger a
        brief reverse to escape corners and tight spots.
        """
        # If we already committed to a direction, keep it.
        if self._committed_turn == 0:
            # Fresh decision: compare clearance with random bias.
            bias = random.uniform(-self._random_bias, self._random_bias)
            if (clearance_left_m + bias) >= (clearance_right_m - bias):
                self._committed_turn = 1  # left
            else:
                self._committed_turn = -1  # right
            self._turn_frames = 0

        self._turn_frames += 1

        # If we've been turning for too long, back up to escape.
        if self._turn_frames > self._reverse_after:
            self._reverse_remaining = self._reverse_frames
            self._turn_frames = 0
            return PolicyDecision(
                left_duty=-self._reverse_duty,
                right_duty=-self._reverse_duty,
                reason=f"reverse escape (stuck {self._reverse_after} frames)",
                state=WanderState.REVERSING,
            )

        # Escalate duty if we've been stuck turning for a while.
        if self._turn_frames > self._turn_escalate_after:
            duty = self._turn_escalate_duty
            esc = " ESC"
        else:
            duty = self._turn_duty
            esc = ""

        if self._committed_turn > 0:
            return PolicyDecision(
                left_duty=-duty,
                right_duty=duty,
                reason=f"turn left{esc} (L={clearance_left_m:.2f} R={clearance_right_m:.2f})",
                state=WanderState.TURNING,
            )
        return PolicyDecision(
            left_duty=duty,
            right_duty=-duty,
            reason=f"turn right{esc} (L={clearance_left_m:.2f} R={clearance_right_m:.2f})",
            state=WanderState.TURNING,
        )

    @staticmethod
    def stopped(reason: str = "stopped") -> PolicyDecision:
        """Convenience: produce a full-stop decision."""
        return PolicyDecision(
            left_duty=0,
            right_duty=0,
            reason=reason,
            state=WanderState.STOPPED,
        )
