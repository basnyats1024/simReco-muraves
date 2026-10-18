# simReco_muraves/cli/compare_golden_routes.py

"""Check that both golden routes produce the same output.

Golden-track output can be produced two ways:

    ClusteredData -> tracking in memory -> golden
    ClusteredData -> TrackedData -> golden

They share a record builder but reach it through different code: the
tracked route reconstructs the selected state from ROOT rather than
holding it in memory. This tool runs both from one clustered file and
compares every common branch, entry by entry.

    muraves-simCompare <clustered_root_file>
    muraves-simCompare --golden left.root right.root

The second form compares two golden files that already exist.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import uproot

from simReco_muraves.golden_tracking.tracked_pipeline import (
    run_golden_tracking_from_tracked,
)


TREE_NAME = "Tracks4ptree"
TRACKED_TREE_NAME = "TrackedData"
CLUSTERED_TREE_NAME = "ClusteredData"

DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_MAX_REPORT = 5


# ----------------------------------------------------------------------
# Value comparison
# ----------------------------------------------------------------------

def values_equal(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two branch values, including jagged ones.

    Exact by default. NaN compares equal to NaN, because a branch that
    is NaN in both routes agrees even though NaN != NaN.
    """
    if isinstance(left, list) or isinstance(right, list):
        if not (isinstance(left, list) and isinstance(right, list)):
            return False

        if len(left) != len(right):
            return False

        return all(
            values_equal(a, b, rtol=rtol, atol=atol)
            for a, b in zip(left, right)
        )

    if isinstance(left, bool) or isinstance(right, bool):
        return left == right

    if isinstance(left, float) or isinstance(right, float):
        left_value = float(left)
        right_value = float(right)

        if math.isnan(left_value) and math.isnan(right_value):
            return True

        if math.isinf(left_value) or math.isinf(right_value):
            return left_value == right_value

        if rtol == 0.0 and atol == 0.0:
            return left_value == right_value

        return abs(left_value - right_value) <= (
            atol + rtol * abs(right_value)
        )

    return left == right


# ----------------------------------------------------------------------
# Tree comparison
# ----------------------------------------------------------------------

class BranchMismatch:

    def __init__(
        self,
        branch: str,
        entry: int,
        left: Any,
        right: Any,
    ) -> None:
        self.branch = branch
        self.entry = entry
        self.left = left
        self.right = right

    def describe(self) -> str:
        return (
            f"    entry {self.entry}: "
            f"left={self.left!r} right={self.right!r}"
        )


def compare_branch(
    left_tree,
    right_tree,
    branch: str,
    *,
    entries: int,
    chunk_size: int,
    rtol: float,
    atol: float,
    max_report: int,
) -> list[BranchMismatch]:

    mismatches: list[BranchMismatch] = []

    for start in range(0, entries, chunk_size):
        stop = min(start + chunk_size, entries)

        left_values = left_tree[branch].array(
            entry_start=start,
            entry_stop=stop,
            library="ak",
        ).tolist()

        right_values = right_tree[branch].array(
            entry_start=start,
            entry_stop=stop,
            library="ak",
        ).tolist()

        for offset, (left_value, right_value) in enumerate(
            zip(left_values, right_values)
        ):
            if values_equal(
                left_value,
                right_value,
                rtol=rtol,
                atol=atol,
            ):
                continue

            mismatches.append(
                BranchMismatch(
                    branch,
                    start + offset,
                    left_value,
                    right_value,
                )
            )

            if len(mismatches) >= max_report:
                return mismatches

    return mismatches


def compare_golden_files(
    left_path: Path,
    right_path: Path,
    *,
    left_label: str,
    right_label: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    rtol: float = 0.0,
    atol: float = 0.0,
    max_report: int = DEFAULT_MAX_REPORT,
) -> bool:
    """Compare two golden ROOT files. Returns True when they agree."""

    print()
    print("==================================================")
    print("Golden route comparison")
    print("==================================================")
    print(f"{left_label:<10}: {left_path}")
    print(f"{right_label:<10}: {right_path}")

    if rtol or atol:
        print(f"tolerance : rtol={rtol} atol={atol}")
    else:
        print("tolerance : exact")

    agreed = True

    with uproot.open(left_path) as left_file, uproot.open(
        right_path
    ) as right_file:

        left_tree = left_file[TREE_NAME]
        right_tree = right_file[TREE_NAME]

        left_entries = left_tree.num_entries
        right_entries = right_tree.num_entries

        print()
        print("Entries")
        print("--------------------------------------------------")
        print(f"  {left_label:<10}: {left_entries:,}")
        print(f"  {right_label:<10}: {right_entries:,}")

        if left_entries != right_entries:
            print("  ENTRY COUNT MISMATCH")
            print()
            print(
                "  Differing entry counts mean the two routes "
                "selected different events; branch comparison is "
                "not meaningful until that is resolved."
            )
            return False

        left_branches = set(left_tree.keys())
        right_branches = set(right_tree.keys())

        only_left = sorted(left_branches - right_branches)
        only_right = sorted(right_branches - left_branches)
        common = sorted(left_branches & right_branches)

        print()
        print("Branches")
        print("--------------------------------------------------")
        print(f"  common            : {len(common)}")
        print(f"  only in {left_label:<10}: {len(only_left)}")
        print(f"  only in {right_label:<10}: {len(only_right)}")

        if only_left:
            agreed = False
            for name in only_left:
                print(f"    < {name}")

        if only_right:
            agreed = False
            for name in only_right:
                print(f"    > {name}")

        if left_entries == 0:
            print()
            print(
                "  Both files are empty; nothing to compare. "
                "Zero-entry output is valid, but this run proves "
                "nothing about route equivalence."
            )
            return agreed

        print()
        print("Values")
        print("--------------------------------------------------")

        mismatched_branches: list[str] = []

        for branch in common:
            mismatches = compare_branch(
                left_tree,
                right_tree,
                branch,
                entries=left_entries,
                chunk_size=chunk_size,
                rtol=rtol,
                atol=atol,
                max_report=max_report,
            )

            if not mismatches:
                continue

            agreed = False
            mismatched_branches.append(branch)

            print(f"  MISMATCH {branch}")

            for mismatch in mismatches:
                print(mismatch.describe())

        print()

        if mismatched_branches:
            print(
                f"  {len(mismatched_branches)} of {len(common)} "
                "branches differ."
            )
        else:
            print(
                f"  All {len(common)} common branches agree "
                f"across {left_entries:,} entries."
            )

    return agreed


# ----------------------------------------------------------------------
# Running both routes
# ----------------------------------------------------------------------

def classify(path: Path) -> str:
    with uproot.open(path) as root_file:
        if TRACKED_TREE_NAME in root_file:
            return "tracked"

        if CLUSTERED_TREE_NAME in root_file:
            return "clustered"

        if TREE_NAME in root_file:
            return "golden"

    return "unknown"


def run_both_routes(
    clustered_file: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Produce golden output by each route into work_dir."""

    # Imported here so that --golden mode works even if the tracking
    # dependencies are unavailable.
    from simReco_muraves.config.reco_config import get_reco_config
    from simReco_muraves.golden_tracking.production_pipeline import (
        run_golden_tracking_output,
    )
    from simReco_muraves.io.clustered_root_reader import load_clusters
    from simReco_muraves.io.tracking_root_writer import (
        build_tracked_event_record,
        write_tracked_root,
    )
    from simReco_muraves.tracking.event_builder_adapter import (
        build_tracking_events,
    )
    from simReco_muraves.tracking.tracking_pipeline_adapter import (
        run_tracking_pipeline,
    )
    from simReco_muraves.cli.run_gTrks import (
        PROXIMITY_CUT_XY,
        PROXIMITY_CUT_XZ,
        SIGMA_Y,
        SIGMA_Z,
        X_POS,
        Y_ADD,
        Z_ADD,
    )

    work_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Running both routes")
    print("--------------------------------------------------")
    print(f"  input    : {clustered_file}")
    print(f"  work dir : {work_dir}")

    clusters = load_clusters(str(clustered_file))

    if not clusters:
        raise RuntimeError(
            f"No clusters loaded from {clustered_file}"
        )

    events = build_tracking_events(clusters)
    configuration = get_reco_config()

    results = run_tracking_pipeline(
        clusters,
        x_pos=X_POS,
        z_add=Z_ADD,
        y_add=Y_ADD,
        sigma_z=SIGMA_Z,
        sigma_y=SIGMA_Y,
        proximity_cut_xz=PROXIMITY_CUT_XZ,
        proximity_cut_xy=PROXIMITY_CUT_XY,
        tracking_cfg=configuration["tracking"],
    )

    if len(events) != len(results):
        raise RuntimeError(
            "Tracking event/result count mismatch: "
            f"events={len(events)}, results={len(results)}"
        )

    print(f"  clusters : {len(clusters):,}")
    print(f"  events   : {len(events):,}")

    # Route A: straight from clusters.
    golden_from_clustered = work_dir / "golden_from_clustered.root"

    run_golden_tracking_output(
        events=events,
        results=results,
        output_path=golden_from_clustered,
        tree_name=TREE_NAME,
    )

    print(f"  wrote    : {golden_from_clustered.name}")

    # Route B: via persisted TrackedData.
    tracked_file = work_dir / "intermediate_tracked.root"

    records = [
        build_tracked_event_record(event, result)
        for event, result in zip(events, results)
    ]

    write_tracked_root(str(tracked_file), records)
    print(f"  wrote    : {tracked_file.name}")

    golden_from_tracked = work_dir / "golden_from_tracked.root"

    run_golden_tracking_from_tracked(
        tracked_root_file=tracked_file,
        output_file=golden_from_tracked,
    )

    print(f"  wrote    : {golden_from_tracked.name}")

    return golden_from_clustered, golden_from_tracked


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="muraves-simCompare",
        description=(
            "Verify that the clustered and tracked golden routes "
            "produce identical output."
        ),
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "ClusteredData ROOT file. Both routes are run from it "
            "and their outputs compared."
        ),
    )

    parser.add_argument(
        "--golden",
        nargs=2,
        type=Path,
        metavar=("LEFT", "RIGHT"),
        default=None,
        help="Compare two golden ROOT files that already exist.",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory for intermediate output "
            "(default: a temporary directory, removed afterwards)."
        ),
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance (default: 0, i.e. exact).",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance (default: 0, i.e. exact).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Entries read per branch at a time "
            f"(default: {DEFAULT_CHUNK_SIZE})."
        ),
    )

    parser.add_argument(
        "--max-report",
        type=int,
        default=DEFAULT_MAX_REPORT,
        help=(
            "Differing entries reported per branch "
            f"(default: {DEFAULT_MAX_REPORT})."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    if arguments.golden is not None and arguments.input_file is not None:
        raise SystemExit(
            "Give either a clustered input file or --golden, "
            "not both."
        )

    if arguments.golden is None and arguments.input_file is None:
        raise SystemExit(
            "Give a clustered input file, or two golden files "
            "with --golden."
        )

    if arguments.golden is not None:
        left_path, right_path = (
            path.expanduser().resolve()
            for path in arguments.golden
        )

        for path in (left_path, right_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        agreed = compare_golden_files(
            left_path,
            right_path,
            left_label="left",
            right_label="right",
            chunk_size=arguments.chunk_size,
            rtol=arguments.rtol,
            atol=arguments.atol,
            max_report=arguments.max_report,
        )

        raise SystemExit(0 if agreed else 1)

    clustered_file = arguments.input_file.expanduser().resolve()

    if not clustered_file.is_file():
        raise FileNotFoundError(clustered_file)

    file_type = classify(clustered_file)

    if file_type != "clustered":
        raise SystemExit(
            f"{clustered_file.name} contains {file_type} data; "
            "this mode needs a ClusteredData file."
        )

    temporary_dir: str | None = None

    if arguments.work_dir is None:
        temporary_dir = tempfile.mkdtemp(prefix="muraves_compare_")
        work_dir = Path(temporary_dir)
    else:
        work_dir = arguments.work_dir.expanduser().resolve()

    try:
        left_path, right_path = run_both_routes(
            clustered_file,
            work_dir,
        )

        agreed = compare_golden_files(
            left_path,
            right_path,
            left_label="clustered",
            right_label="tracked",
            chunk_size=arguments.chunk_size,
            rtol=arguments.rtol,
            atol=arguments.atol,
            max_report=arguments.max_report,
        )
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    print()

    if agreed:
        print("Routes agree.")
    else:
        print("Routes DIFFER; see the report above.")

    print()

    raise SystemExit(0 if agreed else 1)


if __name__ == "__main__":
    main()
