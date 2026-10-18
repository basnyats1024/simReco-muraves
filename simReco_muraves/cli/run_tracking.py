# simReco_muraves/cli/run_tracking.py

"""Phase II: ClusteredData to TrackedData ROOT output.

    muraves-simTrk <clustered_root_file_or_directory> [-o OUTPUT]
    python -m simReco_muraves.cli.run_tracking <clustered_root_file>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simReco_muraves.config.reco_config import get_reco_config
from simReco_muraves.io.clustered_root_reader import load_clusters
from simReco_muraves.io.provenance_tree import attach_provenance
from simReco_muraves.io.tracking_root_writer import (
    TrackedEventRecord,
    build_tracked_event_record,
    write_tracked_root,
)
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


# ==========================================================
# Tracking parameters
# ==========================================================

SIGMA_Z = 0.004
SIGMA_Y = 0.0035

PROXIMITY_CUT_XZ = 0.020
PROXIMITY_CUT_XY = 0.0175


CLUSTERED_PATTERNS = (
    "*_clustered*.root",
    "*clustered*.root",
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="muraves-simTrk",
        description=(
            "Run the MURAVES tracking reconstruction on simulated "
            "clustered data, writing TrackedData ROOT output."
        ),
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "ClusteredData ROOT file, or a directory of them."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output ROOT file. Only valid for a single input file; "
            "defaults to the input name with _clustered replaced by "
            "_tracked."
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
    print("Tracking Configuration")
    print("==================================================")
    print(f"X_POS            = {X_POS}")
    print(f"Z_ADD            = {Z_ADD}")
    print(f"Y_ADD            = {Y_ADD}")
    print(f"SIGMA_Z          = {SIGMA_Z}")
    print(f"SIGMA_Y          = {SIGMA_Y}")
    print(f"PROXIMITY_CUT_XZ = {PROXIMITY_CUT_XZ}")
    print(f"PROXIMITY_CUT_XY = {PROXIMITY_CUT_XY}")
    print("==================================================")
    print()


def build_output_name(input_file: Path) -> Path:
    stem = input_file.stem

    if "_clustered" in stem:
        stem = stem.replace("_clustered", "_tracked")
    else:
        stem = f"{stem}_tracked"

    return input_file.with_name(stem + ".root")


def process_file(
    input_file: Path,
    *,
    output_file: Path | None = None,
    force: bool = False,
) -> None:

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

    print()
    print("--------------------------------------------------")
    print("Processing")
    print("--------------------------------------------------")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print()

    clusters = load_clusters(str(input_file))
    print(f"Clusters loaded : {len(clusters):,}")

    if not clusters:
        raise RuntimeError(f"No clusters loaded from {input_file}")

    tracking_events = build_tracking_events(clusters)
    print(f"Tracking events : {len(tracking_events):,}")

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

    if len(tracking_events) != len(results):
        raise RuntimeError(
            "Tracking event/result count mismatch: "
            f"events={len(tracking_events)}, results={len(results)}"
        )

    records: list[TrackedEventRecord] = [
        build_tracked_event_record(event, result)
        for event, result in zip(tracking_events, results)
    ]

    print(f"Tracked records : {len(records):,}")

    write_tracked_root(str(output_file), records)

    attach_provenance(
        output_file,
        stage="tracking",
        input_path=input_file,
        parameters={
            "x_pos": X_POS,
            "z_add": Z_ADD,
            "y_add": Y_ADD,
            "sigma_z": SIGMA_Z,
            "sigma_y": SIGMA_Y,
            "proximity_cut_xz": PROXIMITY_CUT_XZ,
            "proximity_cut_xy": PROXIMITY_CUT_XY,
            "events": len(tracking_events),
        },
    )

    print()
    print(f"Wrote: {output_file}")
    print()


def get_cluster_files(path: Path) -> list[Path]:
    files: list[Path] = []

    for pattern in CLUSTERED_PATTERNS:
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
            output_file=arguments.output,
            force=arguments.force,
        )
        return

    if arguments.output is not None:
        raise SystemExit(
            "--output is only valid for a single input file."
        )

    cluster_files = get_cluster_files(input_path)
    print(f"Found {len(cluster_files):,} clustered ROOT files.")

    if not cluster_files:
        raise RuntimeError("No clustered ROOT files found.")

    for index, file_path in enumerate(cluster_files, start=1):
        print()
        print(f"[{index}/{len(cluster_files)}]")
        process_file(file_path, force=arguments.force)

    print()
    print("==================================================")
    print("Tracking production finished.")
    print("==================================================")
    print()


if __name__ == "__main__":
    main()
