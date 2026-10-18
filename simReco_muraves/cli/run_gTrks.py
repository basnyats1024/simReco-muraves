# simReco_muraves/cli/run_gTrks.py

"""Phase III: golden-track selection from ClusteredData or TrackedData.

    muraves-simGTrk <root_file_or_directory> [--input-type auto]
    python -m simReco_muraves.cli.run_gTrks <root_file>

The input type is detected from the tree present in the file. Passing
--input-type asserts the expected type rather than overriding
detection: a mismatch is an error, not a silent reinterpretation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uproot

from simReco_muraves.config.reco_config import get_reco_config
from simReco_muraves.golden_tracking.production_pipeline import (
    run_golden_tracking_output,
)
from simReco_muraves.golden_tracking.tracked_pipeline import (
    run_golden_tracking_from_tracked,
)
from simReco_muraves.io.clustered_root_reader import load_clusters
from simReco_muraves.io.provenance_tree import attach_provenance
from simReco_muraves.tracking.event_builder_adapter import (
    build_tracking_events,
)
from simReco_muraves.tracking.tracking_pipeline_adapter import (
    run_tracking_pipeline,
)


# ==========================================================
# Geometry (ROSSO working point)
# ==========================================================

X_POS = [-0.265, 0.0, 0.262, 1.475]
Z_ADD = [0.292, 0.251, 0.207, 0.0]
Y_ADD = [0.0, 0.0, 0.0, 0.0]

SIGMA_Z = 0.004
SIGMA_Y = 0.0035

PROXIMITY_CUT_XZ = 0.020
PROXIMITY_CUT_XY = 0.0175

TREE_NAME = "Tracks4ptree"

INPUT_PATTERNS = (
    "*_clustered*.root",
    "*clustered*.root",
    "*_tracked*.root",
    "*tracked*.root",
)

REQUIRED_BRANCHES = {
    "ProcessID",
    "EventID",
    "Theta",
    "Phi",
    "Scattering_xz",
    "Scattering_xy",
    "ChiSquare_xz",
    "ChiSquare_xy",
    "ChiSquare_3p_xz",
    "ChiSquare_3p_xy",
    "TrackClusterEnergy_Z1",
    "TrackClusterEnergy_Z4",
    "TrackClusterEnergy_Y1",
    "TrackClusterEnergy_Y4",
    "ClusterMultiplicity_Z1",
    "NonTrackClusterEnergy_Z1",
}


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="muraves-simGTrk",
        description=(
            "Produce simulated golden-track output from either "
            "ClusteredData (runs tracking) or TrackedData (reuses "
            "persisted tracking state)."
        ),
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="ClusteredData or TrackedData ROOT file, or a directory.",
    )

    parser.add_argument(
        "--input-type",
        choices=("auto", "clustered", "tracked"),
        default="auto",
        help=(
            "Expected input type. The default detects it from the "
            "file; naming a type asserts it and errors on mismatch."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output ROOT file. Only valid for a single input file."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    return parser.parse_args(argv)


def print_configuration() -> None:
    print()
    print("==================================================")
    print("Simulated Golden-Track Reconstruction")
    print("==================================================")
    print(f"X_POS            = {X_POS}")
    print(f"Z_ADD            = {Z_ADD}")
    print(f"Y_ADD            = {Y_ADD}")
    print(f"SIGMA_Z          = {SIGMA_Z}")
    print(f"SIGMA_Y          = {SIGMA_Y}")
    print(f"PROXIMITY_CUT_XZ = {PROXIMITY_CUT_XZ}")
    print(f"PROXIMITY_CUT_XY = {PROXIMITY_CUT_XY}")
    print(f"TREE_NAME        = {TREE_NAME}")
    print("==================================================")
    print()


def classify_input_file(path: Path) -> str:
    with uproot.open(path) as root_file:
        if "TrackedData" in root_file:
            return "tracked"

        if "ClusteredData" in root_file:
            return "clustered"

        if TREE_NAME in root_file:
            return "golden"

    return "unknown"


def resolve_input_type(path: Path, expected: str) -> str:
    """Detected type, checked against an explicitly requested one."""
    detected = classify_input_file(path)

    if expected == "auto":
        return detected

    if detected != expected:
        raise SystemExit(
            f"--input-type {expected} was requested, but "
            f"{path.name} contains {detected} data."
        )

    return detected


def build_output_name(input_file: Path) -> Path:
    stem = input_file.stem

    for marker in ("_clustered", "_tracked"):
        if marker in stem:
            return input_file.with_name(
                stem.replace(marker, "_golden") + ".root"
            )

    return input_file.with_name(f"{stem}_golden.root")


def inspect_output(output_file: Path) -> None:
    with uproot.open(output_file) as root_file:
        tree = root_file[TREE_NAME]

        physical_branches = list(tree.keys())

        counter_branches = [
            name
            for name in physical_branches
            if name.startswith("nNonTrackCluster")
        ]

        logical_branches = [
            name
            for name in physical_branches
            if name not in counter_branches
        ]

        print()
        print("Golden ROOT output inspection")
        print("--------------------------------------------------")
        print(f"Tree              : {TREE_NAME}")
        print(f"Selected entries  : {tree.num_entries:,}")
        print(f"Logical branches  : {len(logical_branches)}")
        print(f"Counter branches  : {len(counter_branches)}")
        print(f"Physical branches : {len(physical_branches)}")

        missing = sorted(REQUIRED_BRANCHES - set(physical_branches))
        print(f"Missing required branches: {missing}")

        if missing:
            raise RuntimeError("Golden ROOT output is incomplete")

        if tree.num_entries > 0:
            entry_stop = min(10, tree.num_entries)

            process_ids = tree["ProcessID"].array(
                library="np",
                entry_stop=entry_stop,
            ).tolist()

            event_ids = tree["EventID"].array(
                library="np",
                entry_stop=entry_stop,
            ).tolist()

            print(f"First ProcessID values: {process_ids}")
            print(f"First EventID values  : {event_ids}")

        print("Simulated golden Tracks4ptree: OK")


def _prepare_output(
    input_file: Path,
    output_file: Path | None,
    force: bool,
) -> Path:
    if output_file is None:
        output_file = build_output_name(input_file)
    else:
        # Resolve so that the overwrite check, the printed path and
        # the recorded provenance all refer to the same location.
        output_file = output_file.expanduser().resolve()

    if output_file.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_file} "
            "(pass --force to overwrite)"
        )

    return output_file


def process_tracked_file(
    input_file: Path,
    *,
    output_file: Path | None = None,
    force: bool = False,
) -> None:

    output_file = _prepare_output(input_file, output_file, force)

    print()
    print("--------------------------------------------------")
    print("Processing tracked ROOT")
    print("--------------------------------------------------")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    written_path = run_golden_tracking_from_tracked(
        tracked_root_file=input_file,
        output_file=output_file,
    )

    attach_provenance(
        written_path,
        stage="golden",
        input_path=input_file,
        parameters={
            "route": "tracked",
            "tree_name": TREE_NAME,
        },
    )

    print(f"Wrote: {written_path}")

    inspect_output(Path(written_path))


def process_clustered_file(
    input_file: Path,
    *,
    output_file: Path | None = None,
    force: bool = False,
) -> None:

    output_file = _prepare_output(input_file, output_file, force)

    print()
    print("--------------------------------------------------")
    print("Processing clustered ROOT")
    print("--------------------------------------------------")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    clusters = load_clusters(str(input_file))
    print(f"Clusters loaded : {len(clusters):,}")

    if not clusters:
        raise RuntimeError(f"No clusters loaded from {input_file}")

    events = build_tracking_events(clusters)
    print(f"Tracking events : {len(events):,}")

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

    print(f"Tracking results: {len(results):,}")

    if len(events) != len(results):
        raise RuntimeError(
            "Tracking event/result count mismatch: "
            f"events={len(events)}, results={len(results)}"
        )

    written_path = run_golden_tracking_output(
        events=events,
        results=results,
        output_path=output_file,
        tree_name=TREE_NAME,
    )

    attach_provenance(
        written_path,
        stage="golden",
        input_path=input_file,
        parameters={
            "route": "clustered",
            "tree_name": TREE_NAME,
            "x_pos": X_POS,
            "z_add": Z_ADD,
            "y_add": Y_ADD,
            "sigma_z": SIGMA_Z,
            "sigma_y": SIGMA_Y,
            "proximity_cut_xz": PROXIMITY_CUT_XZ,
            "proximity_cut_xy": PROXIMITY_CUT_XY,
        },
    )

    print(f"Wrote: {written_path}")
    inspect_output(Path(written_path))


def process_file(
    input_file: Path,
    *,
    expected_type: str = "auto",
    output_file: Path | None = None,
    force: bool = False,
) -> None:

    file_type = resolve_input_type(input_file, expected_type)

    if file_type == "tracked":
        process_tracked_file(
            input_file,
            output_file=output_file,
            force=force,
        )
        return

    if file_type == "clustered":
        process_clustered_file(
            input_file,
            output_file=output_file,
            force=force,
        )
        return

    if file_type == "golden":
        raise RuntimeError("Input is already a Golden ROOT file")

    raise RuntimeError(f"Unknown ROOT format: {input_file}")


def get_input_files(path: Path) -> list[Path]:
    files: list[Path] = []

    for pattern in INPUT_PATTERNS:
        files.extend(path.glob(pattern))

    return sorted(set(files))


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    input_path = arguments.input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print_configuration()

    if input_path.is_file():
        process_file(
            input_path,
            expected_type=arguments.input_type,
            output_file=arguments.output,
            force=arguments.force,
        )
        return

    if arguments.output is not None:
        raise SystemExit(
            "--output is only valid for a single input file."
        )

    input_files = get_input_files(input_path)
    print(f"Found {len(input_files):,} input ROOT files.")

    if not input_files:
        raise RuntimeError("No compatible ROOT files found")

    failures: list[tuple[Path, Exception]] = []

    for index, file_path in enumerate(input_files, start=1):
        print()
        print(f"[{index}/{len(input_files)}]")

        try:
            file_type = resolve_input_type(
                file_path,
                arguments.input_type,
            )

            if file_type == "golden":
                print(f"Skipping Golden ROOT: {file_path.name}")
                continue

            if file_type == "unknown":
                print(f"Skipping unknown ROOT: {file_path.name}")
                continue

            process_file(
                file_path,
                expected_type=arguments.input_type,
                force=arguments.force,
            )

        except Exception as error:
            failures.append((file_path, error))
            print(
                f"FAILED: {file_path}: "
                f"{type(error).__name__}: {error}"
            )

    print()
    print("==================================================")
    print("Golden tracking production finished")
    print("==================================================")
    print(f"Successful files: {len(input_files) - len(failures)}")
    print(f"Failed files    : {len(failures)}")

    if failures:
        print()
        print("Failures:")
        for file_path, error in failures:
            print(
                f"  {file_path}: "
                f"{type(error).__name__}: {error}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
