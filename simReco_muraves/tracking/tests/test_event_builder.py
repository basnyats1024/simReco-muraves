# simReco_muraves/tracking/tests/test_event_builder.py

"""Grouping clusters into events, and the tracking pipeline's ordering.

Two contracts matter here. Events are keyed by (ProcessID, Event), not
by Event alone -- the Phase I bookkeeping bug. And the tracking
pipeline returns its results in exactly the order build_tracking_events
produces events, because the entry points zip the two lists together.
"""

from __future__ import annotations

from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.tracking import tracking_pipeline_adapter as pipeline
from simReco_muraves.tracking.event_builder_adapter import (
    STATION_MAP,
    build_tracking_events,
)


def cluster(process_id, event, station, position=0.0):
    return ClusterRecord(
        process_id=process_id,
        event=event,
        station=station,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=30.0,
        cluster_position=position,
        cluster_size=1,
        cluster_time=1.0,
        cluster_time_spread=0.0,
        strip_ids=[10],
        strip_energies=[30.0],
    )


def identities(events):
    return [(event.process_id, event.event) for event in events]


# ----------------------------------------------------------------------
# Station map
# ----------------------------------------------------------------------

def test_station_map_covers_the_eight_planes():
    assert sorted(STATION_MAP) == [0, 1, 2, 3, 8, 9, 10, 11]
    assert sorted(STATION_MAP.values()) == [
        "y1", "y2", "y3", "y4", "z1", "z2", "z3", "z4",
    ]


def test_stations_0_to_3_are_y_and_8_to_11_are_z():
    assert [STATION_MAP[s] for s in (0, 1, 2, 3)] == ["y1", "y2", "y3", "y4"]
    assert [STATION_MAP[s] for s in (8, 9, 10, 11)] == ["z1", "z2", "z3", "z4"]


# ----------------------------------------------------------------------
# Event building
# ----------------------------------------------------------------------

def test_each_cluster_lands_on_its_plane():
    events = build_tracking_events([cluster(1, 5, s) for s in STATION_MAP])

    assert len(events) == 1
    assert events[0].n_clusters == 8

    for station, plane in STATION_MAP.items():
        assert [c.station for c in getattr(events[0], plane)] == [station]


def test_same_event_number_in_different_processes_stays_separate():
    """The Phase I bookkeeping bug: Event alone is not unique."""
    events = build_tracking_events([cluster(1, 5, 0), cluster(2, 5, 0)])

    assert identities(events) == [(1, 5), (2, 5)]


def test_events_come_out_in_first_seen_order():
    events = build_tracking_events(
        [cluster(2, 1, 0), cluster(1, 1, 0), cluster(2, 1, 8)]
    )

    assert identities(events) == [(2, 1), (1, 1)]
    assert events[0].n_clusters == 2


def test_cluster_order_within_a_plane_is_preserved():
    """Tracking refers to clusters by position in these lists."""
    events = build_tracking_events(
        [cluster(1, 5, 8, position) for position in (0.3, 0.1, 0.2)]
    )

    assert [c.cluster_position for c in events[0].z1] == [0.3, 0.1, 0.2]


def test_clusters_from_unknown_stations_are_dropped():
    """Pinned current behaviour: dropped silently, but the event exists."""
    events = build_tracking_events([cluster(1, 5, 4), cluster(1, 5, 0)])

    assert len(events) == 1
    assert events[0].n_clusters == 1

    only_unknown = build_tracking_events([cluster(1, 6, 99)])

    assert identities(only_unknown) == [(1, 6)]
    assert only_unknown[0].n_clusters == 0


def test_no_clusters_gives_no_events():
    assert build_tracking_events([]) == []


# ----------------------------------------------------------------------
# Pipeline ordering
# ----------------------------------------------------------------------

GEOMETRY = dict(
    x_pos=[-0.265, 0.0, 0.262, 1.475],
    z_add=[0.292, 0.251, 0.207, 0.0],
    y_add=[0.0, 0.0, 0.0, 0.0],
    sigma_z=0.004,
    sigma_y=0.0035,
    proximity_cut_xz=0.020,
    proximity_cut_xy=0.0175,
    tracking_cfg={},
)


def test_pipeline_results_follow_event_order(monkeypatch):
    """The entry points zip events with results; the orders must match."""
    seen = []

    def fake_reconstruct(event, **configuration):
        seen.append((event.process_id, event.event, configuration))
        return (event.process_id, event.event)

    monkeypatch.setattr(pipeline, "reconstruct_event", fake_reconstruct)

    clusters = [
        cluster(2, 1, 0),
        cluster(1, 1, 0),
        cluster(2, 1, 8),
        cluster(1, 7, 0),
    ]

    results = pipeline.run_tracking_pipeline(clusters, **GEOMETRY)

    assert results == identities(build_tracking_events(clusters))
    assert len(seen) == len(results)


def test_pipeline_passes_the_configuration_through(monkeypatch):
    received = {}

    def fake_reconstruct(event, **configuration):
        received.update(configuration)
        return None

    monkeypatch.setattr(pipeline, "reconstruct_event", fake_reconstruct)

    pipeline.run_tracking_pipeline([cluster(1, 1, 0)], **GEOMETRY)

    assert received == GEOMETRY
