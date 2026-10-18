# Manual clustered-data tracking pipeline check

from __future__ import annotations

from simReco_muraves.io.clustered_root_reader import load_clusters



from simReco_muraves.config.reco_config import (
    get_reco_config,
)

from simReco_muraves.tracking.tracking_pipeline_adapter import (
    run_tracking_pipeline,
)


INPUT_FILE = "/Users/sbasnet/Documents/MURAVES_stuff/digiWork/srcDigi/dataFiles/simData/TMP/musimData_run4_clustered_part000.root"


def main():

    print(f"\nLoading: {INPUT_FILE}")

    clusters = load_clusters(INPUT_FILE)

    print(f"Clusters loaded: {len(clusters):,}")

    cfg = get_reco_config()

    results = run_tracking_pipeline(
        clusters,
        x_pos=[0.265, 0.000, -0.262, -1.475],
        z_add=[0.0, 0.0, 0.0, 0.0],
        y_add=[0.0, 0.0, 0.0, 0.0],
        sigma_z=0.004,
        sigma_y=0.0035,
        proximity_cut_xz=0.020,
        proximity_cut_xy=0.0175,
        tracking_cfg=cfg["tracking"],
    )

    print(f"\nTracking events: {len(results):,}")

    n_evt_xz = sum(
        r.ntracks_3p_xz > 0
        for r in results
    )

    n_evt_xy = sum(
        r.ntracks_3p_xy > 0
        for r in results
    )

    mean_ntracks_xz = (
        sum(r.ntracks_3p_xz for r in results)
        / len(results)
    )

    mean_ntracks_xy = (
        sum(r.ntracks_3p_xy for r in results)
        / len(results)
    )

    print("\nSummary")
    print("-------")

    print(
        f"Events with XZ tracks : "
        f"{n_evt_xz:,} "
        f"({100*n_evt_xz/len(results):.2f}%)"
    )

    print(
        f"Events with XY tracks : "
        f"{n_evt_xy:,} "
        f"({100*n_evt_xy/len(results):.2f}%)"
    )

    print(
        f"Mean Ntracks_3p_xz : "
        f"{mean_ntracks_xz:.4f}"
    )

    print(
        f"Mean Ntracks_3p_xy : "
        f"{mean_ntracks_xy:.4f}"
    )

    print("\nFirst 10 tracking results\n")

    for result in results[:10]:

        print(
            result.process_id,
            result.event,
            result.ntracks_3p_xz,
            result.ntracks_3p_xy,
            result.theta_3p,
            result.phi_3p,
        )


if __name__ == "__main__":
    main()
