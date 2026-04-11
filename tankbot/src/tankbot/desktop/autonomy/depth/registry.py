"""Backend registry so the benchmark can name candidates on the CLI."""

from __future__ import annotations

from collections.abc import Callable

from tankbot.desktop.autonomy.depth.base import DepthEstimator

_Factory = Callable[[], DepthEstimator]


def _depth_anything_v2_factory() -> DepthEstimator:
    from tankbot.desktop.autonomy.depth.depth_anything_v2 import DepthAnythingV2Estimator

    return DepthAnythingV2Estimator()


def _depth_pro_factory() -> DepthEstimator:
    from tankbot.desktop.autonomy.depth.depth_pro import DepthProEstimator

    return DepthProEstimator()


def _unidepth_factory() -> DepthEstimator:
    from tankbot.desktop.autonomy.depth.unidepth import UniDepthEstimator

    return UniDepthEstimator()


KNOWN_BACKENDS: dict[str, _Factory] = {
    "depth_anything_v2": _depth_anything_v2_factory,
    "depth_pro": _depth_pro_factory,
    "unidepth": _unidepth_factory,
}


def list_backends() -> list[str]:
    return sorted(KNOWN_BACKENDS.keys())


def build_estimator(name: str) -> DepthEstimator:
    try:
        factory = KNOWN_BACKENDS[name]
    except KeyError as exc:
        msg = f"unknown depth backend '{name}'; known: {', '.join(list_backends())}"
        raise ValueError(msg) from exc
    return factory()
