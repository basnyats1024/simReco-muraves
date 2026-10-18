from __future__ import annotations

from pathlib import Path

import uproot

from simReco_muraves.models.cluster_record import ClusterRecord


CLUSTER_TREE_NAME = "ClusteredData"

CLUSTER_BRANCHES = (
    "ProcessID",
    "Event",
    "Station",
    "ClusterID",
    "ClusterMultiplicity",
    "ClusterEnergy",
    "ClusterPosition",
    "ClusterSize",
    "ClusterTime",
    "ClusterTimeSpread",
    "StripIDs",
    "StripEnergies",
)


def load_clusters(
    root_file: str | Path,
    *,
    tree_name: str = CLUSTER_TREE_NAME,
) -> list[ClusterRecord]:
    """Read reconstructed clusters from a ROOT file.

    Parameters
    ----------
    root_file:
        ROOT file containing the clustered-data tree.
    tree_name:
        Name of the clustered-data tree.

    Returns
    -------
    list[ClusterRecord]
        Cluster records in their original ROOT entry order.
    """
    path = Path(root_file)

    if not path.is_file():
        raise FileNotFoundError(
            f"Clustered ROOT file does not exist: {path}"
        )

    with uproot.open(path) as input_file:
        if tree_name not in input_file:
            available = ", ".join(input_file.keys())

            raise KeyError(
                f"Tree {tree_name!r} was not found in {path}. "
                f"Available objects: {available}"
            )

        tree = input_file[tree_name]

        missing = [
            branch
            for branch in CLUSTER_BRANCHES
            if branch not in tree.keys()
        ]

        if missing:
            raise KeyError(
                "Missing clustered-data branches: "
                + ", ".join(missing)
            )

        arrays = tree.arrays(
            list(CLUSTER_BRANCHES),
            library="ak",
        )

    clusters: list[ClusterRecord] = []

    for index in range(len(arrays["Event"])):
        clusters.append(
            ClusterRecord(
                process_id=int(arrays["ProcessID"][index]),
                event=int(arrays["Event"][index]),
                station=int(arrays["Station"][index]),
                cluster_id=int(arrays["ClusterID"][index]),
                cluster_multiplicity=int(
                    arrays["ClusterMultiplicity"][index]
                ),
                cluster_energy=float(
                    arrays["ClusterEnergy"][index]
                ),
                cluster_position=float(
                    arrays["ClusterPosition"][index]
                ),
                cluster_size=int(
                    arrays["ClusterSize"][index]
                ),
                cluster_time=float(
                    arrays["ClusterTime"][index]
                ),
                cluster_time_spread=float(
                    arrays["ClusterTimeSpread"][index]
                ),
                strip_ids=[
                    int(value)
                    for value in arrays["StripIDs"][index]
                ],
                strip_energies=[
                    float(value)
                    for value in arrays["StripEnergies"][index]
                ],
            )
        )

    return clusters
