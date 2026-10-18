# scripts/manual_checks/check_tracked_to_golden.py

"""Smoke-check the tracked -> golden route on a single file.

Uses the same record builder as production, so this script cannot
drift from the pipeline it is meant to be checking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simReco_muraves.golden_tracking.golden_pipeline import (
    process_golden_events,
)
from simReco_muraves.golden_tracking.golden_record_builder import (
    build_golden_record,
)
from simReco_muraves.golden_tracking.production_pipeline import (
    write_golden_records,
)
from simReco_muraves.io.tracked_root_reader import load_tracked_events


DEFAULT_TRACKED_FILE = (
    "validation_data/raw_100_tracked_part000.root"
)

DEFAULT_OUTPUT_FILE = (
    "validation_data/test_golden_from_tracked.root"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the tracked -> golden projection on one "
            "TrackedData file and report what happened."
        )
    )

    parser.add_argument(
        "tracked_file",
        nargs="?",
        default=DEFAULT_TRACKED_FILE,
        type=Path,
        help="TrackedData ROOT file to project.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        type=Path,
        help="Golden ROOT file to write.",
    )

    return parser.parse_args()


def report_texp_null(tracked_events) -> None:
    counts: dict[int, int] = {}

    for _, result in tracked_events:
        flag = result.best_track_4p_is_texp_null_xz
        counts[flag] = counts.get(flag, 0) + 1

    labels = {
        -1: "no 4p promotion",
        0: "valid Texp",
        1: "fell back to first candidate",
    }

    print()
    print("Plane-4 candidate selection (XZ)")
    print("-" * 50)

    for flag in sorted(counts):
        print(
            f"  {labels.get(flag, f'unknown ({flag})'):<32}"
            f"{counts[flag]:>6}"
        )


def main() -> None:
    arguments = parse_arguments()

    tracked_events = load_tracked_events(
        str(arguments.tracked_file)
    )

    print(f"Input          : {arguments.tracked_file}")
    print(f"Entries loaded : {len(tracked_events)}")

    batch = process_golden_events(
        tracked_events,
        lambda tracked_event: build_golden_record(
            planes=tracked_event[0],
            result=tracked_event[1],
        ),
    )

    diagnostics = batch.diagnostics

    print()
    print("Golden projection")
    print("-" * 50)
    print(f"  input events       : {diagnostics.input_events}")
    print(f"  selected           : {diagnostics.selected_events}")
    print(f"  rejected           : {diagnostics.rejected_events}")
    print(f"  consistency errors : {diagnostics.consistency_errors}")

    if diagnostics.rejection_counts:
        print()
        print("  rejections by reason:")
        for reason, count in sorted(
            diagnostics.rejection_counts.items()
        ):
            print(f"    {reason:<40}{count:>6}")

    if diagnostics.consistency_error_counts:
        print()
        print("  consistency errors by reason:")
        for reason, count in sorted(
            diagnostics.consistency_error_counts.items()
        ):
            print(f"    {reason:<40}{count:>6}")

    report_texp_null(tracked_events)

    records = [
        {
            branch_name: batch.columns[branch_name][entry_index]
            for branch_name in batch.columns
        }
        for entry_index in range(batch.entry_count)
    ]

    output_path = write_golden_records(
        records=records,
        diagnostics=diagnostics,
        output_path=arguments.output,
    )

    print()
    print(f"Wrote: {output_path}")

    if not records:
        print("No selected events; nothing further to show.")
        return

    first = records[0]

    print()
    print("First selected record")
    print("-" * 50)

    for branch_name in (
        "ProcessID",
        "EventID",
        "Theta",
        "Phi",
        "ChiSquare_xz",
        "ChiSquare_xy",
        "ChiSquare_3p_xz",
        "ChiSquare_3p_xy",
        "TrackClusterEnergy_Z1",
        "TrackClusterEnergy_Z4",
    ):
        print(f"  {branch_name:<26}{first[branch_name]}")


if __name__ == "__main__":
    main()
