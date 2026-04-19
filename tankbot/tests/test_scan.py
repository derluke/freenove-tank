from __future__ import annotations

import numpy as np

from tankbot.desktop.autonomy.reactive.scan import ScanConfig, extract_scan


def test_extract_scan_uses_robust_bin_percentile() -> None:
    """A single near outlier in one angular bin must not own the scan."""
    cfg = ScanConfig(n_bins=181, range_percentile=35.0)

    # Dominant surface at ~2 m straight ahead, all inside the scan slice.
    wall = np.array(
        [[2.0 + 0.01 * i, 0.0, 0.10] for i in range(-10, 11)],
        dtype=np.float32,
    )
    # One spuriously-near point in the same angular bin.
    outlier = np.array([[0.45, 0.0, 0.10]], dtype=np.float32)
    points = np.concatenate([wall, outlier], axis=0)

    scan = extract_scan(points, cfg)
    center_bin = int(np.argmin(np.abs(scan.angles)))

    assert scan.ranges[center_bin] > 1.5, (
        f"near outlier dominated scan bin: range={scan.ranges[center_bin]:.3f} m"
    )
