# tracking/event_builder.py

from __future__ import annotations

from ..models.cluster_record import ClusterRecord
from ..models.tracking_event import TrackingEvent


STATION_MAP = {
    0: "y1",
    1: "y2",
    2: "y3",
    3: "y4",
    8: "z1",
    9: "z2",
    10: "z3",
    11: "z4",
}


def build_tracking_events(
    clusters: list[ClusterRecord],
) -> list[TrackingEvent]:

    grouped: dict[tuple[int, int], TrackingEvent] = {}

    for cluster in clusters:

        key = (
            cluster.process_id,
            cluster.event,
        )

        if key not in grouped:

            grouped[key] = TrackingEvent(
                process_id=cluster.process_id,
                event=cluster.event,
            )

        tracking_event = grouped[key]

        plane_name = STATION_MAP.get(cluster.station)

        if plane_name is None:
            continue

        getattr(tracking_event, plane_name).append(cluster)

    return list(grouped.values())
