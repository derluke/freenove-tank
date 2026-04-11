"""UniDepth (v2) backend.

UniDepth is metric monocular and lives between Depth Anything V2 and Depth
Pro on the size / speed axis. It is *not* on PyPI; the upstream install
instructions are ``pip install git+https://github.com/lpiccinelli-eth/UniDepth``.
Weights auto-download through HuggingFace Hub.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from tankbot.desktop.autonomy.depth.base import (
    BackendUnavailable,
    DepthEstimator,
    DepthResult,
)

DEFAULT_CHECKPOINT = "lpiccinelli/unidepth-v2-vitl14"
CHECKPOINT_ENV = "TANKBOT_UNIDEPTH_CHECKPOINT"


class UniDepthEstimator(DepthEstimator):
    name = "unidepth"

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self._checkpoint = checkpoint or os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
        self._device = device
        self._model: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                f"missing dependency: {exc.name}",
                install_hint="uv sync --extra desktop",
            ) from exc

        try:
            from unidepth.models import UniDepthV2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                "unidepth package not installed",
                install_hint=("uv pip install 'git+https://github.com/lpiccinelli-eth/UniDepth'"),
            ) from exc

        try:
            model = UniDepthV2.from_pretrained(self._checkpoint)
        except Exception as exc:
            raise BackendUnavailable(
                self.name,
                f"failed to load checkpoint {self._checkpoint!r}: {exc}",
                install_hint="check network + HF cache permissions",
            ) from exc

        model.eval()
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailable(
                self.name,
                "CUDA requested but torch.cuda.is_available() is False",
                install_hint="install a CUDA-enabled torch build",
            )
        model.to(self._device)
        self._torch = torch
        self._model = model

    def infer(self, bgr_image: np.ndarray) -> DepthResult:
        if self._model is None or self._torch is None:
            msg = "UniDepthEstimator.infer called before load()"
            raise RuntimeError(msg)
        torch = self._torch

        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            msg = f"expected HxWx3 BGR image, got shape {bgr_image.shape}"
            raise ValueError(msg)

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # UniDepth wants CHW float tensor in [0, 255] range.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self._device)

        with torch.inference_mode():
            predictions = self._model.infer(tensor)

        depth_tensor = predictions["depth"]  # (1, 1, H, W) metric meters
        if depth_tensor.ndim == 4:
            depth_tensor = depth_tensor[0, 0]
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor[0]

        if depth_tensor.shape != (h, w):
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor[None, None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]

        depth = depth_tensor.to(torch.float32).detach().cpu().numpy()

        return DepthResult(
            depth_m=depth,
            valid_mask=None,
            input_shape=(h, w),
            backend=self.name,
        )

    def unload(self) -> None:
        if self._model is None:
            return
        torch = self._torch
        self._model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
