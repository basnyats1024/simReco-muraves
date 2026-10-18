from __future__ import annotations

from types import SimpleNamespace

import pytest

uproot = pytest.importorskip("uproot")

from simReco_muraves.golden_tracking.production_pipeline import (
    reconstruct_tracking_events,
    run_golden_from_clusters,
)
from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.models.tracking_event import TrackingEvent


def cluster(event_id: int, station: int, energy: float):
    return ClusterRecord(
        process_id=1,
        event=event_id,
        station=station,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=energy,
        cluster_position=0.0,
        cluster_size=1,
        cluster_time=100.0,
        cluster_time_spread=0.0,
        strip_ids=[0],
        strip_energies=[energy],
    )


def event_clusters(event_id: int):
    return [
        cluster(event_id, 0, 20.0),
        cluster(event_id, 1, 21.0),
        cluster(event_id, 2, 22.0),
        cluster(event_id, 3, 23.0),
        cluster(event_id, 8, 10.0),
        cluster(event_id, 9, 11.0),
        cluster(event_id, 10, 12.0),
        cluster(event_id, 11, 13.0),
    ]


def tracks():
    return SimpleNamespace(
        cluster_index_1=[0],
        cluster_index_2=[0],
        cluster_index_3=[0],
        cluster_indices_4=[[0.0]],
        ScatteringAngles=[[1.25]],
    )


def result_for(event: TrackingEvent, selected: bool = True):
    return SimpleNamespace(
        process_id=event.process_id,
        event=event.event,
        tracks_xz=tracks(),
        tracks_xy=tracks(),
        best_track_xz_index=0,
        best_track_xy_index=0,
        track_3p_of_4p_index_xz=0,
        track_3p_of_4p_index_xy=0,
        track_4p_index_xz=0,
        track_4p_index_xy=0,
        best_track_3p_chi_square_xz=1.0 if selected else -1.0,
        best_track_3p_chi_square_xy=2.0,
        best_track_4p_chi_square_xz=3.0,
        best_track_4p_chi_square_xy=4.0,
        theta_4p=22.5,
        phi_4p=180.0,
    )


def test_reconstructs_events_in_input_order():
    events = [
        TrackingEvent(process_id=1, event=20),
        TrackingEvent(process_id=1, event=10),
    ]

    seen = []

    def reconstruct(event):
        seen.append(event.event)
        return result_for(event)

    results = reconstruct_tracking_events(events, reconstruct)

    assert seen == [20, 10]
    assert [result.event for result in results] == [20, 10]


def test_rejects_mismatched_reconstruction_identity():
    event = TrackingEvent(process_id=1, event=10)

    def reconstruct(input_event):
        result = result_for(input_event)
        result.event = 11
        return result

    with pytest.raises(ValueError, match="mismatched event identity"):
        reconstruct_tracking_events([event], reconstruct)


def test_builds_events_reconstructs_and_writes_root(tmp_path):
    clusters = event_clusters(20) + event_clusters(10)
    output_path = tmp_path / "golden.root"

    run_golden_from_clusters(
        clusters=clusters,
        reconstruct_event=result_for,
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 2
        assert tree["EventID"].array(library="np").tolist() == [20, 10]
        assert tree["TrackClusterEnergy_Z1"].array(
            library="np"
        ).tolist() == [10.0, 10.0]


def test_writes_zero_entry_root_when_reconstruction_rejects_all(tmp_path):
    output_path = tmp_path / "empty.root"

    run_golden_from_clusters(
        clusters=event_clusters(10),
        reconstruct_event=lambda event: result_for(event, selected=False),
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 0
        assert "TrackClusterEnergy_Z1" in tree


def test_supports_custom_tree_name(tmp_path):
    output_path = tmp_path / "custom.root"

    run_golden_from_clusters(
        clusters=event_clusters(10),
        reconstruct_event=result_for,
        output_path=output_path,
        tree_name="GoldenTracks",
    )

    with uproot.open(output_path) as root_file:
        assert "GoldenTracks" in root_file
        assert "Tracks4ptree" not in root_file


def test_unknown_stations_do_not_create_selected_clusters(tmp_path):
    clusters = event_clusters(10)
    clusters.append(cluster(10, 99, 999.0))
    output_path = tmp_path / "golden.root"

    run_golden_from_clusters(
        clusters=clusters,
        reconstruct_event=result_for,
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 1
        assert tree["ClusterMultiplicity_Z1"].array(
            library="np"
        ).tolist() == [1]
