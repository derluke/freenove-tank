"""2D ICP scan matching for frame-to-frame odometry.

Per the Phase 2 plan: "ICP against the previous frame with gyro as a
rotation prior, returns yaw_rad from gyro and xy_m from the matched
translation."

The matcher accepts two `Scan2D` instances and computes the rigid 2D
transform (rotation + translation) that best aligns the source scan to
the reference scan. It optionally accepts a gyro-derived rotation prior
which, when available, constrains the ICP rotation to near the prior
and lets the optimizer focus on translation.

Without a gyro prior (Phase 2 without IMU), the ICP estimates both
rotation and translation from the scan geometry alone. This works for
most motions but is weakest during pure in-place turns on featureless
walls — exactly the case the plan flags.

Confidence is based on the mean point-to-point residual after
convergence. Low confidence triggers health degradation in the
`ScanMatchPoseSource`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tankbot.desktop.autonomy.reactive.scan import Scan2D


@dataclass(frozen=True)
class MatchResult:
    """Result of an ICP scan match."""

    dx: float
    """Translation along robot x (forward), meters."""

    dy: float
    """Translation along robot y (left), meters."""

    dyaw: float
    """Rotation (radians, CCW positive)."""

    mean_residual: float
    """Mean point-to-point distance after convergence (meters).
    Lower is better. Typical good match: < 0.05 m."""

    converged: bool
    """True if ICP converged before hitting max_iterations."""

    n_correspondences: int
    """Number of point pairs used in the final iteration."""

    iterations: int
    """Number of ICP iterations actually performed."""


@dataclass(frozen=True)
class ICPConfig:
    """Tunable parameters for 2D ICP."""

    max_iterations: int = 30
    """Maximum ICP iterations."""

    convergence_threshold: float = 1e-4
    """Stop when the change in mean residual between iterations is below
    this (meters)."""

    max_correspondence_dist: float = 0.50
    """Reject point pairs farther apart than this (meters). Rejects
    outliers from dynamic objects or scan edges."""

    min_valid_bins: int = 10
    """Minimum number of valid bins in both scans to attempt matching.
    Below this, the scan is too sparse for reliable matching. Lowered
    from 20 after the Phase 0c `straight_2m` sweep — monocular depth
    regularly produces scans with 10-30 valid bins, and 20 was rejecting
    usable matches."""

    rotation_prior_weight: float = 20.0
    """When a gyro rotation prior is provided, this weight (relative to
    unit data weight) penalizes deviation from the prior. At 20.0, the
    ICP essentially locks rotation to the gyro and solves for translation
    only — appropriate since gyro is far more reliable than monocular
    depth scan geometry for rotation."""


def _build_kd_pairs(
    src: np.ndarray,
    ref: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest-neighbor correspondences from src to ref.

    Returns matched (src_pts, ref_pts) arrays, filtered by max_dist.
    Uses brute-force O(N*M) since scan point counts are small (~100-200).
    """
    # src: (N, 2), ref: (M, 2)
    # Compute pairwise distances.
    diff = src[:, np.newaxis, :] - ref[np.newaxis, :, :]  # (N, M, 2)
    dists = np.sqrt((diff * diff).sum(axis=-1))  # (N, M)

    # For each source point, find the nearest reference point.
    nearest_idx = dists.argmin(axis=1)  # (N,)
    nearest_dist = dists[np.arange(src.shape[0]), nearest_idx]  # (N,)

    # Filter by max distance.
    valid = nearest_dist <= max_dist
    return src[valid], ref[nearest_idx[valid]]


def _estimate_transform(
    src: np.ndarray,
    ref: np.ndarray,
    rotation_prior: float | None = None,
    prior_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Estimate rigid 2D transform (dx, dy, dyaw) aligning src to ref.

    Uses SVD-based closed-form solution. When a rotation prior is
    provided, adds a soft constraint pulling dyaw toward the prior.

    Returns (dx, dy, dyaw).
    """
    n = src.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0

    # Centroids.
    src_mean = src.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    src_c = src - src_mean
    ref_c = ref - ref_mean

    # Cross-covariance matrix.
    H = src_c.T @ ref_c  # (2, 2)

    # If we have a rotation prior and nonzero weight, augment H.
    if rotation_prior is not None and prior_weight > 0.0:
        # Add a virtual correspondence that pulls toward the prior rotation.
        cos_p = np.cos(rotation_prior)
        sin_p = np.sin(rotation_prior)
        # The prior says R should be [[cos, -sin], [sin, cos]].
        # Adding weight * R^T to H biases the SVD toward this rotation.
        R_prior = np.array([[cos_p, sin_p], [-sin_p, cos_p]], dtype=np.float64)
        H = H + prior_weight * n * R_prior

    U, _, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1).
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])
    R = Vt.T @ S @ U.T

    dyaw = float(np.arctan2(R[1, 0], R[0, 0]))
    t = ref_mean - R @ src_mean
    return float(t[0]), float(t[1]), dyaw


def match_scans(
    source: Scan2D,
    reference: Scan2D,
    *,
    config: ICPConfig | None = None,
    rotation_prior: float | None = None,
) -> MatchResult:
    """Run ICP to align `source` scan to `reference` scan.

    Parameters
    ----------
    source
        The newer scan (current frame).
    reference
        The older scan (previous frame).
    config
        ICP parameters. Uses defaults if None.
    rotation_prior
        Optional gyro-derived rotation between the two scans (radians,
        CCW positive). When provided, the ICP uses it as a soft
        constraint on the rotation component.

    Returns
    -------
    MatchResult
        The estimated rigid transform from source frame to reference
        frame, plus quality metrics.
    """
    cfg = config or ICPConfig()

    # Convert scans to cartesian point sets.
    src_pts = source.to_cartesian().astype(np.float64)
    ref_pts = reference.to_cartesian().astype(np.float64)

    # Check minimum point counts.
    if src_pts.shape[0] < cfg.min_valid_bins or ref_pts.shape[0] < cfg.min_valid_bins:
        return MatchResult(
            dx=0.0,
            dy=0.0,
            dyaw=0.0,
            mean_residual=float("inf"),
            converged=False,
            n_correspondences=0,
            iterations=0,
        )

    # Accumulated transform.
    total_dx = 0.0
    total_dy = 0.0
    total_dyaw = 0.0
    current_src = src_pts.copy()

    prev_residual = float("inf")
    converged = False
    final_n_corr = 0
    n_iters = 0

    for _it in range(1, cfg.max_iterations + 1):
        n_iters = _it
        # Find correspondences.
        matched_src, matched_ref = _build_kd_pairs(current_src, ref_pts, cfg.max_correspondence_dist)

        if matched_src.shape[0] < 3:
            break

        final_n_corr = matched_src.shape[0]

        # Compute residual remaining rotation prior: subtract what
        # we've already rotated by.
        remaining_prior = None
        if rotation_prior is not None:
            remaining_prior = rotation_prior - total_dyaw

        # Estimate incremental transform.
        dx, dy, dyaw = _estimate_transform(
            matched_src,
            matched_ref,
            rotation_prior=remaining_prior,
            prior_weight=cfg.rotation_prior_weight if rotation_prior is not None else 0.0,
        )

        # Apply incremental transform to source points.
        cos_d = np.cos(dyaw)
        sin_d = np.sin(dyaw)
        R = np.array([[cos_d, -sin_d], [sin_d, cos_d]])
        current_src = (R @ current_src.T).T + np.array([dx, dy])

        # Accumulate.
        cos_t = np.cos(total_dyaw)
        sin_t = np.sin(total_dyaw)
        total_dx += cos_t * dx - sin_t * dy
        total_dy += sin_t * dx + cos_t * dy
        total_dyaw += dyaw

        # Check convergence on mean residual.
        diffs = current_src[: matched_src.shape[0]] - matched_ref
        residual = float(np.sqrt((diffs * diffs).sum(axis=-1)).mean()) if matched_src.shape[0] > 0 else float("inf")

        if abs(prev_residual - residual) < cfg.convergence_threshold:
            converged = True
            prev_residual = residual
            break
        prev_residual = residual

    return MatchResult(
        dx=total_dx,
        dy=total_dy,
        dyaw=total_dyaw,
        mean_residual=prev_residual,
        converged=converged,
        n_correspondences=final_n_corr,
        iterations=n_iters,
    )
