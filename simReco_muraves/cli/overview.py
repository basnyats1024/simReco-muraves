# simReco_muraves/cli/overview.py

"""Overview of the simulation reconstruction commands.

    muraves-sim                     list the chain and its commands
    muraves-sim --provenance FILE   show how a ROOT file was produced
    muraves-sim --version
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simReco_muraves.provenance import (
    SCHEMA_VERSION,
    package_version,
)


COMMANDS = (
    (
        "muraves-simCluster",
        "hits -> clusters",
        "Phase I",
    ),
    (
        "muraves-simTrk",
        "clusters -> tracks",
        "Phase II",
    ),
    (
        "muraves-simGTrk",
        "clusters | tracks -> golden tracks",
        "Phase III",
    ),
    (
        "muraves-simCompare",
        "check both golden routes agree",
        "validation",
    ),
)

CHAIN = (
    "Simulated HitData -> ClusteredData -> TrackedData "
    "-> Tracks4ptree"
)


def print_overview() -> None:
    print()
    print(
        "MURAVES simulation reconstruction  "
        f"(simreco-muraves {package_version()})"
    )
    print()
    print(f"  {CHAIN}")
    print()
    print("Commands")

    for name, purpose, phase in COMMANDS:
        print(f"  {name:<20} {purpose:<36} {phase}")

    print()
    print("Run any command with --help for its options.")
    print()
    print(
        "  muraves-sim --provenance FILE   "
        "show how a ROOT file was produced"
    )
    print()
    print(f"Golden output schema version: {SCHEMA_VERSION}")
    print()


def print_provenance(path: Path) -> int:
    # Imported here so that the plain overview works even if the ROOT
    # dependencies are unavailable.
    from simReco_muraves.io.provenance_tree import read_chain
    from simReco_muraves.provenance import summarize_chain

    if not path.is_file():
        raise FileNotFoundError(path)

    chain = read_chain(path)

    print()
    print(f"Provenance: {path}")
    print("=" * 60)
    print()
    print(summarize_chain(chain))
    print()

    return 0 if chain else 1


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="muraves-sim",
        description=(
            "Overview of the MURAVES simulation reconstruction "
            "commands."
        ),
    )

    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        metavar="FILE",
        help="Show the recorded lineage of a ROOT output file.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed package version and exit.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    if arguments.version:
        print(package_version())
        return

    if arguments.provenance is not None:
        status = print_provenance(
            arguments.provenance.expanduser().resolve()
        )

        if status:
            raise SystemExit(status)

        return

    print_overview()


if __name__ == "__main__":
    main()
