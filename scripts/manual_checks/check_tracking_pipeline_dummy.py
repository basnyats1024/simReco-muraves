# test_tracking_pipeline.py

from simReco_muraves.models.cluster_record import ClusterRecord

from simReco_muraves.tracking.event_builder_adapter import (
    build_tracking_events,
)

from simReco_muraves.tracking.tracking_pipeline_adapter import (
    run_tracking_pipeline,
)

from simReco_muraves.config.reco_config import get_reco_config

cfg = get_reco_config()

'''
print("\nCONFIG TYPE:", type(cfg))
print("CONFIG CONTENTS:")
print(cfg)
print()

raise SystemExit

'''

clusters = [
    ClusterRecord(
        process_id=1,
        event=1,
        station=8,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=10.0,
        cluster_position=0.10,
        cluster_size=1,
        cluster_time=100.0,
        cluster_time_spread=0.0,
        strip_ids=[1],
        strip_energies=[10.0],
    ),
    ClusterRecord(
        process_id=1,
        event=1,
        station=9,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=11.0,
        cluster_position=0.11,
        cluster_size=1,
        cluster_time=101.0,
        cluster_time_spread=0.0,
        strip_ids=[2],
        strip_energies=[11.0],
    ),
    ClusterRecord(
        process_id=1,
        event=1,
        station=10,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=12.0,
        cluster_position=0.12,
        cluster_size=1,
        cluster_time=102.0,
        cluster_time_spread=0.0,
        strip_ids=[3],
        strip_energies=[12.0],
    ),
    ClusterRecord(
        process_id=1,
        event=1,
        station=0,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=10.0,
        cluster_position=0.20,
        cluster_size=1,
        cluster_time=100.0,
        cluster_time_spread=0.0,
        strip_ids=[4],
        strip_energies=[10.0],
    ),
    ClusterRecord(
        process_id=1,
        event=1,
        station=1,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=11.0,
        cluster_position=0.21,
        cluster_size=1,
        cluster_time=101.0,
        cluster_time_spread=0.0,
        strip_ids=[5],
        strip_energies=[11.0],
    ),
    ClusterRecord(
        process_id=1,
        event=1,
        station=2,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=12.0,
        cluster_position=0.22,
        cluster_size=1,
        cluster_time=102.0,
        cluster_time_spread=0.0,
        strip_ids=[6],
        strip_energies=[12.0],
    ),
]

events = build_tracking_events(clusters)

print(f"Tracking events: {len(events)}")

results = run_tracking_pipeline(
    clusters,
    x_pos=[0.0, 1.0, 2.0, 3.0],
    z_add=[0.0, 0.0, 0.0, 0.0],
    y_add=[0.0, 0.0, 0.0, 0.0],
    sigma_z=0.004,
    sigma_y=0.0035,
    proximity_cut_xz=0.020,
    proximity_cut_xy=0.0175,
    tracking_cfg=cfg["tracking"],
)

for result in results:
    print(result)
