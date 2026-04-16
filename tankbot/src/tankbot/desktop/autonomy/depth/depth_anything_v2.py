"""Depth Anything V2 (metric indoor) backend.

Uses the HuggingFace `transformers` port
(`depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`) so integration is a
few lines and weights auto-download.

Metric: yes (indoor split — appropriate for our indoor robot use case).
Measured on RTX 4070 Ti SUPER at 640x480 serial inference (2026-04-11):
Small ~56 FPS (245 MiB), Base ~23 FPS (615 MiB), Large ~8 FPS (1711 MiB).
Base is the Phase 0b selection — see docs/autonomy_perception_redesign.md.
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

# Default checkpoint — Base is the Phase 0b-selected variant (23 Hz on
# RTX 4070 Ti SUPER). Large is higher quality but fails the 10 Hz gate.
# Override via TANKBOT_DEPTH_ANYTHING_V2_CHECKPOINT env var if needed.
DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
CHECKPOINT_ENV = "TANKBOT_DEPTH_ANYTHING_V2_CHECKPOINT"


class DepthAnythingV2Estimator(DepthEstimator):
    name = "depth_anything_v2"

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self._checkpoint = checkpoint or os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
        self._device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                f"missing dependency: {exc.name}",
                install_hint="uv sync --extra desktop",
            ) from exc

        try:
            processor = AutoImageProcessor.from_pretrained(self._checkpoint)
            model = AutoModelForDepthEstimation.from_pretrained(self._checkpoint)
        except Exception as exc:  # huggingface_hub errors vary by version
            raise BackendUnavailable(
                self.name,
                f"failed to load checkpoint {self._checkpoint!r}: {exc}",
                install_hint="check network + HF_TOKEN if the model is gated",
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
        self._processor = processor
        self._model = model

    def infer(self, bgr_image: np.ndarray) -> DepthResult:
        if self._model is None or self._processor is None or self._torch is None:
            msg = "DepthAnythingV2Estimator.infer called before load()"
            raise RuntimeError(msg)
        torch = self._torch

        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            msg = f"expected HxWx3 BGR image, got shape {bgr_image.shape}"
            raise ValueError(msg)

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # transformers processors accept numpy arrays directly.
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted = outputs.predicted_depth  # (1, H', W') metric meters

        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)  # (1, 1, H', W')
        # Resize to original resolution on GPU (cheaper than host round-trip).
        resized = torch.nn.functional.interpolate(predicted, size=(h, w), mode="bilinear", align_corners=False)
        depth = resized[0, 0].to(torch.float32).detach().cpu().numpy()

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
        self._processor = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
