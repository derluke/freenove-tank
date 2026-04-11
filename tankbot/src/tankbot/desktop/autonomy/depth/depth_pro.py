"""Depth Pro backend.

Uses the HuggingFace port `apple/DepthPro-hf`, which exposes Apple's Depth
Pro through the standard `transformers` depth-estimation API. Depth Pro is
natively metric and targets high accuracy at the cost of latency (~1B
params) — we expect this backend to be the highest-quality and the one most
at risk of failing the Phase 0b gate.
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

DEFAULT_CHECKPOINT = "apple/DepthPro-hf"
CHECKPOINT_ENV = "TANKBOT_DEPTH_PRO_CHECKPOINT"


class DepthProEstimator(DepthEstimator):
    name = "depth_pro"

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
        except Exception as exc:
            raise BackendUnavailable(
                self.name,
                f"failed to load checkpoint {self._checkpoint!r}: {exc}",
                install_hint=(
                    "Depth Pro on HF requires transformers>=4.45. Upgrade transformers or set HF_TOKEN if gated."
                ),
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
            msg = "DepthProEstimator.infer called before load()"
            raise RuntimeError(msg)
        torch = self._torch

        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            msg = f"expected HxWx3 BGR image, got shape {bgr_image.shape}"
            raise ValueError(msg)

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Depth Pro's HF head returns either `predicted_depth` or a post-
        # processed depth. Prefer the processor's own post-processing if
        # available so metric scaling stays consistent across transformers
        # versions.
        depth_tensor: Any = None
        if hasattr(self._processor, "post_process_depth_estimation"):
            try:
                post = self._processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
                if post and isinstance(post, list) and "predicted_depth" in post[0]:
                    depth_tensor = post[0]["predicted_depth"]
            except Exception:  # fall through to raw predicted_depth
                depth_tensor = None

        if depth_tensor is None:
            depth_tensor = getattr(outputs, "predicted_depth", None)
            if depth_tensor is None:
                msg = "Depth Pro outputs did not expose predicted_depth"
                raise RuntimeError(msg)
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(1)
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor, size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]

        depth = depth_tensor.to(torch.float32).detach().cpu().numpy()
        if depth.ndim != 2:
            depth = np.squeeze(depth)

        return DepthResult(
            depth_m=depth.astype(np.float32),
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
