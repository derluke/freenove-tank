"""Dataset writer / reader regression tests.

The main regression guard here is the append-mode bug that silently
concatenated two recording sessions into one directory during Phase 0c.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankbot.desktop.autonomy.dataset import (
    FRAME_LOG_NAME,
    DatasetWriter,
    iter_frame_records,
)


def test_open_refuses_existing_frame_log(tmp_path: Path) -> None:
    out = tmp_path / "run1"
    writer = DatasetWriter(out, source="test")
    writer.open()
    # Write one frame so the log is non-empty.
    writer.write_frame(frame_index=0, t_monotonic=0.0, t_wall=0.0)
    writer.close()

    # Second writer on the same dir should refuse before touching anything.
    writer2 = DatasetWriter(out, source="test")
    with pytest.raises(FileExistsError, match="refusing to append"):
        writer2.open()


def test_open_into_fresh_dir_succeeds(tmp_path: Path) -> None:
    out = tmp_path / "fresh"
    writer = DatasetWriter(out, source="test")
    writer.open()
    writer.write_frame(frame_index=0, t_monotonic=0.0, t_wall=0.0)
    writer.close()
    assert (out / FRAME_LOG_NAME).exists()


def test_iter_frame_records_monotonic(phase0c_dir: Path) -> None:
    """Any recording in phase0c should have monotonically increasing frame indices
    and timestamps — this is the invariant the append-mode bug violated."""
    any_run = next(
        (d for d in sorted(phase0c_dir.iterdir()) if (d / FRAME_LOG_NAME).exists()),
        None,
    )
    if any_run is None:
        pytest.skip("no phase0c recordings available")

    prev_idx = -1
    prev_t = float("-inf")
    for rec in iter_frame_records(any_run):
        assert rec.index > prev_idx, (
            f"{any_run.name}: frame index went backwards at {rec.index}"
        )
        assert rec.t_monotonic >= prev_t, (
            f"{any_run.name}: t_monotonic went backwards at frame {rec.index}"
        )
        prev_idx = rec.index
        prev_t = rec.t_monotonic
