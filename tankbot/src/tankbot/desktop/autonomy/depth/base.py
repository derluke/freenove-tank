"""Contract shared by every monocular depth backend.

Kept deliberately small — the benchmark is the only current consumer. A
downstream reactive layer can import `DepthResult` without pulling any
backend-specific dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BackendUnavailable(RuntimeError):
    """Raised by `DepthEstimator.load` when required deps / weights are missing.

    The benchmark catches this and reports a skip with the included hint, so a
    missing backend never aborts the whole run.
    """

    def __init__(self, backend: str, detail: str, install_hint: str | None = None) -> None:
        self.backend = backend
        self.install_hint = install_hint
        msg = f"{backend}: {detail}"
        if install_hint:
            msg += f" (try: {install_hint})"
        super().__init__(msg)


@dataclass(frozen=True)
class DepthResult:
    """Single-frame metric depth output.

    `depth_m` is always HxW float32 meters. Distances that the backend
    considers unreliable (sky, reflection, far clip) are represented by
    `valid_mask=False` at that pixel. `valid_mask=None` means "backend did
    not report a validity mask; treat every pixel as valid."
    """

    depth_m: np.ndarray
    valid_mask: np.ndarray | None
    input_shape: tuple[int, int]  # (H, W) of the image that was fed in
    backend: str

    def __post_init__(self) -> None:
        if self.depth_m.dtype != np.float32:
            msg = f"depth_m must be float32, got {self.depth_m.dtype}"
            raise ValueError(msg)
        if self.depth_m.ndim != 2:
            msg = f"depth_m must be HxW, got shape {self.depth_m.shape}"
            raise ValueError(msg)
        if self.valid_mask is not None:
            if self.valid_mask.shape != self.depth_m.shape:
                msg = f"valid_mask shape {self.valid_mask.shape} does not match depth_m shape {self.depth_m.shape}"
                raise ValueError(msg)
            if self.valid_mask.dtype != np.bool_:
                msg = f"valid_mask must be bool, got {self.valid_mask.dtype}"
                raise ValueError(msg)


class DepthEstimator(ABC):
    """Pluggable monocular metric depth backend.

    Contract:
      - `load()` is idempotent. It may download weights on first call.
      - `infer()` takes a BGR `uint8` HxWx3 image (OpenCV convention, matches
        the rest of the repo) and returns a `DepthResult` whose `depth_m`
        matches the input H, W exactly — upsampling is the backend's job.
      - `unload()` releases GPU memory. Safe to call when not loaded.
      - `name` is a short human-readable identifier used in reports.
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load weights + move model to the target device.

        Raises `BackendUnavailable` if deps or weights cannot be obtained.
        """

    @abstractmethod
    def infer(self, bgr_image: np.ndarray) -> DepthResult:
        """Run a single-frame inference. Must be safe to call repeatedly."""

    def unload(self) -> None:  # pragma: no cover - default no-op
        """Release GPU memory. Override when the backend holds heavy state."""
        return None
