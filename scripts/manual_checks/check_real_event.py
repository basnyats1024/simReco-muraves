from __future__ import annotations

from simReco_muraves.io.root_reader import RootReader
from simReco_muraves.clustering.sim_clusterizer import aggregate_bar_hits
from simReco_muraves.clustering.sim_clusterizer import PoissonDigitizer
from simReco_muraves.clustering.sim_clusterizer import build_planes
from simReco_muraves.clustering.sim_clusterizer import SimClusterizer


def main():

    event_id = 0

    print("\n==============================")
    print("READ EVENT")
    print("==============================")

    reader = RootReader("/Users/sbasnet/Documents/MURAVES_stuff/digiWork/srcDigi/dataFiles/simData/musimData_run4.root")

    raw_hits = reader.read_event(event_id)

    print(f"Event ID           : {event_id}")
    print(f"Raw hits           : {len(raw_hits)}")

    print("\n==============================")
    print("BAR AGGREGATION")
    print("==============================")

    bar_hits = aggregate_bar_hits(raw_hits)

    print(f"Aggregated bars    : {len(bar_hits)}")

    print("\n==============================")
    print("DIGITIZATION")
    print("==============================")

    digitizer = PoissonDigitizer(
        pe_per_mev=8.5,
        rng_seed=42,
    )

    digitized_hits = digitizer.digitize(bar_hits)

    print(f"Digitized bars     : {len(digitized_hits)}")

    print("\n==============================")
    print("PLANE BUILDING")
    print("==============================")

    planes = build_planes(digitized_hits)

    print(f"Number of planes   : {len(planes)}")

    print("\n==============================")
    print("CLUSTERIZATION")
    print("==============================")

    clusterizer = SimClusterizer()

    total_clusters = 0

    planes = sorted(
        planes,
        key=lambda p: p.station,
    )

    for plane in planes:

        clusters = clusterizer.cluster_plane(plane)

        n_clusters = len(
            clusters.ClustersEnergy
        )

        total_clusters += n_clusters

        print(f"\nStation {plane.station}")
        print(f"  Occupied strips : {plane.occupied_strips}")
        print(f"  Clusters found  : {n_clusters}")

        max_to_print = min(5, n_clusters)

        for idx in range(max_to_print):

            energy = clusters.ClustersEnergy[idx]
            position = clusters.ClustersPositions[idx]
            size = clusters.ClustersSize[idx]

            print(f"\n    Cluster {idx}")
            print(f"      Energy   : {energy:.2f} nPE")
            print(f"      Position : {position:.5f}")
            print(f"      Size     : {size}")
            print(f"      Strips   : {clusters.StripsID[idx]}")

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print(f"Total planes   : {len(planes)}")
    print(f"Total clusters : {total_clusters}")


if __name__ == "__main__":
    main()
