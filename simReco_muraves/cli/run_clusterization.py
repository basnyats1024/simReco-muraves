# simReco_muraves/cli/run_clusterization.py

"""Phase I: simulated hits to ClusteredData ROOT output.

    muraves-simCluster <hit_root_file> [-o OUTPUT_DIR]
    python -m simReco_muraves.cli.run_clusterization <hit_root_file>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simReco_muraves.clustering.sim_clusterizer import (
    PoissonDigitizer,
    SimClusterizer,
    aggregate_bar_hits,
    build_planes,
    cluster_collection_to_records,
)
from simReco_muraves.io.provenance_tree import (
    attach_provenance_to_many,
)
from simReco_muraves.io.root_reader import RootReader
from simReco_muraves.io.root_writer import RootWriter


# Detector-response defaults. These reproduce the configuration used for
# the Phase I-III validation samples; change them only deliberately,
# since the cluster energy scale depends directly on PE_PER_MEV.
DEFAULT_PE_PER_MEV = 8.5
DEFAULT_RNG_SEED = 42
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_PROGRESS_EVERY = 25_000


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="muraves-simCluster",
        description=(
            "Aggregate simulated Geant4 hits, digitize them and run "
            "the MURAVES clusterization, writing ClusteredData ROOT "
            "chunks."
        ),
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="ROOT file containing the simulated HitData tree.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the ClusteredData chunks "
            "(default: alongside the input file)."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Chunk filename prefix "
            "(default: <input stem>_clustered)."
        ),
    )

    parser.add_argument(
        "--pe-per-mev",
        type=float,
        default=DEFAULT_PE_PER_MEV,
        help=(
            "Photoelectron yield per MeV used by the digitizer "
            f"(default: {DEFAULT_PE_PER_MEV})."
        ),
    )

    parser.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=(
            "Seed for the Poisson digitizer "
            f"(default: {DEFAULT_RNG_SEED})."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Clusters per output ROOT chunk "
            f"(default: {DEFAULT_CHUNK_SIZE})."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Print progress every N events "
            f"(default: {DEFAULT_PROGRESS_EVERY})."
        ),
    )

    return parser.parse_args(argv)


def print_configuration(arguments: argparse.Namespace) -> None:
    print()
    print("==================================================")
    print("Simulated Clusterization")
    print("==================================================")
    print(f"Input          = {arguments.input_file}")
    print(f"Output dir     = {arguments.output_dir}")
    print(f"Output prefix  = {arguments.output_prefix}")
    print(f"PE per MeV     = {arguments.pe_per_mev}")
    print(f"RNG seed       = {arguments.rng_seed}")
    print(f"Chunk size     = {arguments.chunk_size:,}")
    print("==================================================")
    print()


def run_clusterization(arguments: argparse.Namespace) -> None:
    reader = RootReader(str(arguments.input_file))

    writer = RootWriter(
        output_prefix=arguments.output_prefix,
        output_dir=str(arguments.output_dir),
        chunk_size=arguments.chunk_size,
    )

    digitizer = PoissonDigitizer(
        pe_per_mev=arguments.pe_per_mev,
        rng_seed=arguments.rng_seed,
    )

    clusterizer = SimClusterizer()

    event_keys = reader.get_event_keys()

    events_with_clusters = 0
    events_without_clusters = 0
    total_clusters = 0

    for event_number, (process_id, event_id) in enumerate(event_keys):

        if (
            arguments.progress_every > 0
            and event_number % arguments.progress_every == 0
        ):
            print(
                f"Processing {event_number:,}/{len(event_keys):,}"
            )

        raw_hits = reader.read_event(process_id, event_id)

        if len(raw_hits) == 0:
            events_without_clusters += 1
            continue

        bar_hits = aggregate_bar_hits(raw_hits)
        digitized_hits = digitizer.digitize(bar_hits)
        planes = build_planes(digitized_hits)

        event_has_clusters = False

        for plane in planes:
            cluster_collection = clusterizer.cluster_plane(plane)

            records = cluster_collection_to_records(
                event=event_id,
                station=plane.station,
                cluster_collection=cluster_collection,
                plane=plane,
            )

            if len(records) > 0:
                event_has_clusters = True
                total_clusters += len(records)

            writer.extend(records)

        if event_has_clusters:
            events_with_clusters += 1
        else:
            events_without_clusters += 1

    writer.close()

    print()
    print("==================================================")
    print("Production Summary")
    print("==================================================")
    print(f"Unique event keys processed : {len(event_keys):,}")
    print(f"Events with clusters        : {events_with_clusters:,}")
    print(f"Events without clusters     : {events_without_clusters:,}")
    print(f"Total clusters              : {total_clusters:,}")

    if len(event_keys) > 0:
        efficiency = 100.0 * events_with_clusters / len(event_keys)
        clusters_per_event = total_clusters / len(event_keys)

        print(f"Cluster efficiency          : {efficiency:.2f}%")
        print(f"Clusters/event              : {clusters_per_event:.2f}")

    chunk_files = sorted(
        arguments.output_dir.glob(
            f"{arguments.output_prefix}_part*.root"
        )
    )

    if chunk_files:
        attach_provenance_to_many(
            chunk_files,
            stage="clusterization",
            input_path=arguments.input_file,
            parameters={
                "pe_per_mev": arguments.pe_per_mev,
                "rng_seed": arguments.rng_seed,
                "chunk_size": arguments.chunk_size,
                "output_prefix": arguments.output_prefix,
                "events_processed": len(event_keys),
                "total_clusters": total_clusters,
            },
        )

    print()
    print("Finished clusterization.")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    input_file = arguments.input_file.expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    arguments.input_file = input_file

    if arguments.output_dir is None:
        arguments.output_dir = input_file.parent
    else:
        arguments.output_dir = (
            arguments.output_dir.expanduser().resolve()
        )

    if arguments.output_prefix is None:
        arguments.output_prefix = f"{input_file.stem}_clustered"

    print_configuration(arguments)
    run_clusterization(arguments)


if __name__ == "__main__":
    main()
