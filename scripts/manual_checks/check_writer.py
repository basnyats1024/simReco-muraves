from simReco_muraves.io.root_reader import RootReader
from simReco_muraves.io.root_writer import RootWriter

from simReco_muraves.models.cluster_record import ClusterRecord

print(ClusterRecord)
print(ClusterRecord.__annotations__)

from simReco_muraves.clustering.sim_clusterizer import (
    aggregate_bar_hits,
)

from simReco_muraves.clustering.sim_clusterizer import (
    PoissonDigitizer,
)

from simReco_muraves.clustering.sim_clusterizer import (
    build_planes,
)

from simReco_muraves.clustering.sim_clusterizer import (
    SimClusterizer,
)

from simReco_muraves.clustering.sim_clusterizer import (
    cluster_collection_to_records,
)


def main():

    event_id = 0

    reader = RootReader(
        "/Users/sbasnet/Documents/MURAVES_stuff/digiWork/srcDigi/dataFiles/simData/musimData_run4.root"
    )

    raw_hits = reader.read_event(
        event_id
    )

    bar_hits = aggregate_bar_hits(
        raw_hits
    )

    digitizer = PoissonDigitizer(
        pe_per_mev=8.5,
        rng_seed=42,
    )

    digitized_hits = digitizer.digitize(
        bar_hits
    )

    planes = build_planes(
        digitized_hits
    )

    clusterizer = SimClusterizer()

    writer = RootWriter(
        "event0_clustered.root"
    )

    total_clusters = 0

    for plane in planes:

        clusters = clusterizer.cluster_plane(
            plane
        )

        records = (
            cluster_collection_to_records(
                event=event_id,
                station=plane.station,
                cluster_collection=clusters,
                plane=plane,
            )
        )

        total_clusters += len(records)

        writer.extend(records)

    writer.write()

    print(
        f"Total clusters: {total_clusters}"
    )


if __name__ == "__main__":
    main()
