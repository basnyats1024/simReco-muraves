from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

uproot = pytest.importorskip("uproot")

from simReco_muraves.golden_tracking.production_pipeline import run_golden_reconstruction
from simReco_muraves.models.cluster_record import ClusterRecord


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


def reconstruct(event, selected=True):
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


def test_writes_root_and_default_diagnostics(tmp_path):
    output_path = tmp_path / "golden.root"

    artifacts = run_golden_reconstruction(
        clusters=event_clusters(10),
        reconstruct_event=reconstruct,
        output_path=output_path,
        metadata={"sample": "unit_test"},
    )

    assert artifacts.root_path == output_path
    assert artifacts.diagnostics_path == tmp_path / "golden.root.diagnostics.json"
    assert artifacts.root_path.exists()
    assert artifacts.diagnostics_path.exists()

    with uproot.open(artifacts.root_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 1
        assert tree["EventID"].array(library="np").tolist() == [10]

    payload = json.loads(
        artifacts.diagnostics_path.read_text(encoding="utf-8")
    )
    assert payload["diagnostics"]["input_events"] == 1
    assert payload["diagnostics"]["selected_events"] == 1
    assert payload["metadata"]["sample"] == "unit_test"
    assert payload["metadata"]["tree_name"] == "Tracks4ptree"


def test_writes_custom_diagnostics_path(tmp_path):
    diagnostics_path = tmp_path / "reports" / "phase3.json"

    artifacts = run_golden_reconstruction(
        clusters=event_clusters(10),
        reconstruct_event=reconstruct,
        output_path=tmp_path / "golden.root",
        diagnostics_path=diagnostics_path,
    )

    assert artifacts.diagnostics_path == diagnostics_path
    assert diagnostics_path.exists()


def test_records_zero_entry_rejection(tmp_path):
    artifacts = run_golden_reconstruction(
        clusters=event_clusters(10),
        reconstruct_event=lambda event: reconstruct(event, selected=False),
        output_path=tmp_path / "empty.root",
    )

    with uproot.open(artifacts.root_path) as root_file:
        assert root_file["Tracks4ptree"].num_entries == 0

    payload = json.loads(
        artifacts.diagnostics_path.read_text(encoding="utf-8")
    )
    assert payload["diagnostics"]["input_events"] == 1
    assert payload["diagnostics"]["selected_events"] == 0
    assert payload["diagnostics"]["rejected_events"] == 1
    assert payload["diagnostics"]["rejection_counts"] == {
        "invalid_3p_xz": 1,
    }


def test_preserves_event_order(tmp_path):
    artifacts = run_golden_reconstruction(
        clusters=event_clusters(20) + event_clusters(10),
        reconstruct_event=reconstruct,
        output_path=tmp_path / "golden.root",
    )

    with uproot.open(artifacts.root_path) as root_file:
        event_ids = root_file["Tracks4ptree"]["EventID"].array(
            library="np"
        ).tolist()

    assert event_ids == [20, 10]


def test_records_consistency_error(tmp_path):
    def bad_reconstruction(event):
        result = reconstruct(event)
        result.best_track_xy_index = -1
        return result

    artifacts = run_golden_reconstruction(
        clusters=event_clusters(10),
        reconstruct_event=bad_reconstruction,
        output_path=tmp_path / "invalid.root",
    )

    payload = json.loads(
        artifacts.diagnostics_path.read_text(encoding="utf-8")
    )
    assert payload["diagnostics"]["selected_events"] == 0
    assert payload["diagnostics"]["consistency_errors"] == 1
