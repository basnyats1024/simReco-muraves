# test_tracking_root_writer.py

from simReco_muraves.config.reco_config import get_reco_config

from simReco_muraves.io.tracking_root_writer import (
    TrackedEventRecord,
)

from simReco_muraves.tracking.tracking_pipeline_adapter import (
    run_tracking_pipeline,
)

from simReco_muraves.io.tracking_root_writer import (
    build_tracked_event_record,
)

from simReco_muraves.io.tracking_root_writer import (
    write_tracked_root,
)

from simReco_muraves.io.clustered_root_reader import (
    load_clusters,
)


INPUT_FILE = (
    "/Users/sbasnet/Documents/MURAVES_stuff/"
    "digiWork/srcDigi/dataFiles/simData/TMP/"
    "musimData_run4_clustered_part000.root"
)

OUTPUT_FILE = "test_tracked.root"


def main():

    print(f"\nLoading: {INPUT_FILE}")

    clusters = load_clusters(INPUT_FILE)

    print(f"Clusters loaded: {len(clusters):,}")

    cfg = get_reco_config()

    results = run_tracking_pipeline(
        clusters,
        x_pos = [-0.265, 0.0, 0.262, 1.475],
        z_add = [0.292, 0.251, 0.207, 0.0],
        y_add=[0.0, 0.0, 0.0, 0.0],
        sigma_z=0.004,
        sigma_y=0.0035,
        proximity_cut_xz=0.020,
        proximity_cut_xy=0.0175,
        tracking_cfg=cfg["tracking"],
    )

    print(f"Tracking results: {len(results):,}")

    records: list[TrackedEventRecord] = []

    for result in results:
        records.append(
            build_tracked_event_record(result)
        )

    print(f"Tracked records: {len(records):,}")

    write_tracked_root(
        OUTPUT_FILE,
        records,
    )

    print(f"\nWrote ROOT file:\n{OUTPUT_FILE}")

    import uproot

    tree = uproot.open(OUTPUT_FILE)["TrackedData"]

    print("\nNew branches check:\n")

    for branch in [
        "TrackPosition_z1",
        "TrackPosition_z2",
        "TrackPosition_z3",
        "TrackEnergy_z1",
        "TrackEnergy_z2",
        "TrackEnergy_z3",
        "TrackTime_z1",
        "TrackTime_z2",
        "TrackTime_z3",
    ]:

        print(branch)

        arr = tree[branch].array()

        print(arr[0])

        print()

if __name__ == "__main__":
    main()
