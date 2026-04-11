"""Monocular metric depth backends for autonomy perception (Phase 0b).

This package exposes a small abstraction (`DepthEstimator`) plus three
candidate backends — Depth Anything V2, Depth Pro, UniDepth — so the Phase 0b
benchmark can measure them against the performance gate documented in
`docs/autonomy_perception_redesign.md` §3.1.1.

Nothing in this package wires into the live control loop. Phase 1 selects the
winning backend and imports it from here.
"""

from __future__ import annotations

from tankbot.desktop.autonomy.depth.base import (
    BackendUnavailable,
    DepthEstimator,
    DepthResult,
)
from tankbot.desktop.autonomy.depth.registry import (
    KNOWN_BACKENDS,
    build_estimator,
    list_backends,
)

__all__ = [
    "KNOWN_BACKENDS",
    "BackendUnavailable",
    "DepthEstimator",
    "DepthResult",
    "build_estimator",
    "list_backends",
]
