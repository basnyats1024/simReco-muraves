# simReco_muraves/test_pipeline.py

from __future__ import annotations

from simReco_muraves.models.raw_hit import RawHit
from simReco_muraves.clustering.sim_clusterizer import aggregate_bar_hits


def load_test_hits():
    return [
        RawHit(
            event=0,
            station=0,
            module=0,
            bar=10,
            edep_mev=2.5,
            time_ns=1.0,
            track_id=1,
            parent_id=0,
            pdg=13,
        ),
        RawHit(
            event=0,
            station=0,
            module=0,
            bar=10,
            edep_mev=0.5,
            time_ns=1.1,
            track_id=2,
            parent_id=1,
            pdg=11,
        ),
        RawHit(
            event=0,
            station=0,
            module=0,
            bar=11,
            edep_mev=1.5,
            time_ns=1.03,
            track_id=1,
            parent_id=0,
            pdg=13,
        ),
    ]


def main():

    print("\n==============================")
    print("PIPELINE SMOKE TEST")
    print("==============================")

    raw_hits = load_test_hits()

    print(f"Raw hits: {len(raw_hits)}")

    bar_hits = aggregate_bar_hits(raw_hits)

    print(f"Aggregated bars: {len(bar_hits)}")

    for i, bar in enumerate(bar_hits):

        print(
            f"[BAR {i}] "
            f"event={bar.event} "
            f"station={bar.station} "
            f"module={bar.module} "
            f"bar={bar.bar} "
            f"strip={bar.strip} "
            f"E={bar.energy_mev:.3f} MeV "
            f"mult={bar.multiplicity} "
            f"dt={bar.dt_ns:.3f} ns"
        )

    print("\nSUCCESS: aggregation stage completed.")


if __name__ == "__main__":
    main()
