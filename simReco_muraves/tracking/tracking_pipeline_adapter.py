# tracking/tracking_pipeline_adapter.py

from __future__ import annotations

from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.models.tracking_results import TrackingResult

from simReco_muraves.tracking.event_builder_adapter import (
    build_tracking_events,
)

from simReco_muraves.tracking.reconstruction_adapter import (
    reconstruct_event,
)


def run_tracking_pipeline(
    clusters: list[ClusterRecord],
    *,
    x_pos: list[float],
    z_add: list[float],
    y_add: list[float],
    sigma_z: float,
    sigma_y: float,
    proximity_cut_xz: float,
    proximity_cut_xy: float,
    tracking_cfg: dict,
) -> list[TrackingResult]:
    """
    Run MURAVES tracking reconstruction on a flat list of ClusterRecord
    objectsngResult per (ProcessID, Event).
    """

    tracking_events = build_tracking_events(clusters)

    results: list[TrackingResult] = []

    for event in tracking_events:

        result = reconstruct_event(
            event=event,
            x_pos=x_pos,
            z_add=z_add,
            y_add=y_add,
            sigma_z=sigma_z,
            sigma_y=sigma_y,
            proximity_cut_xz=proximity_cut_xz,
            proximity_cut_xy=proximity_cut_xy,
            tracking_cfg=tracking_cfg,
        )

        results.append(result)

    return results
