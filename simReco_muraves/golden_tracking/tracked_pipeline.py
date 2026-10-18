"""Golden reconstruction directly from TrackedData ROOT files.

This pipeline reuses persisted cluster payloads and tracking
bookkeeping from TrackedData to generate Golden ROOT output
without re-running tracking reconstruction.

It shares both the event loop (process_golden_events) and the record
builder (build_golden_record) with the clustered route, so the two
produce the same records, the same gate and the same diagnostics keys.
"""

from __future__ import annotations

from pathlib import Path

from .golden_record_builder import build_golden_record
from .golden_pipeline import process_golden_events
from .production_pipeline import write_golden_records

from ..io.tracked_root_reader import load_tracked_events


def run_golden_tracking_from_tracked(
    tracked_root_file: str | Path,
    output_file: str | Path,
    *,
    tree_name: str = "Tracks4ptree",
) -> Path:

    tracked_events = load_tracked_events(
        str(tracked_root_file)
    )

    batch = process_golden_events(
        tracked_events,
        lambda tracked_event: build_golden_record(
            planes=tracked_event[0],
            result=tracked_event[1],
        ),
    )

    records = [
        {
            branch_name: batch.columns[branch_name][entry_index]
            for branch_name in batch.columns
        }
        for entry_index in range(batch.entry_count)
    ]

    return write_golden_records(
        records=records,
        diagnostics=batch.diagnostics,
        output_path=output_file,
        tree_name=tree_name,
    )
