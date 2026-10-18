from __future__ import annotations

from types import SimpleNamespace

import pytest

uproot = pytest.importorskip("uproot")

from simReco_muraves.golden_tracking.production_pipeline import (
    build_result_index,
    run_golden_tracking_output,
)
from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.models.tracking_event import TrackingEvent


def cluster(process_id, event_id, station, energy):
    return ClusterRecord(
        process_id=process_id,
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


def make_event(event_id):
    return TrackingEvent(
        process_id=1,
        event=event_id,
        y1=[cluster(1, event_id, 0, 20.0)],
        y2=[cluster(1, event_id, 1, 21.0)],
        y3=[cluster(1, event_id, 2, 22.0)],
        y4=[cluster(1, event_id, 3, 23.0)],
        z1=[cluster(1, event_id, 8, 10.0)],
        z2=[cluster(1, event_id, 9, 11.0)],
        z3=[cluster(1, event_id, 10, 12.0)],
        z4=[cluster(1, event_id, 11, 13.0)],
    )


def tracks():
    return SimpleNamespace(
        cluster_index_1=[0],
        cluster_index_2=[0],
        cluster_index_3=[0],
        cluster_indices_4=[[0.0]],
        ScatteringAngles=[[1.25]],
    )


def make_result(event_id, selected=True):
    return SimpleNamespace(
        process_id=1,
        event=event_id,
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


def test_builds_result_index():
    result = make_result(10)
    indexed = build_result_index([result])
    assert indexed[(1, 10)] is result


def test_rejects_duplicate_result_identity():
    with pytest.raises(ValueError, match="Duplicate TrackingResult"):
        build_result_index([make_result(10), make_result(10)])


def test_writes_selected_events_in_event_order(tmp_path):
    output_path = tmp_path / "golden.root"

    run_golden_tracking_output(
        events=[make_event(20), make_event(10)],
        results=[make_result(10), make_result(20)],
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 2
        assert tree["EventID"].array(library="np").tolist() == [20, 10]


def test_writes_zero_entry_output_when_all_rejected(tmp_path):
    output_path = tmp_path / "empty.root"

    run_golden_tracking_output(
        events=[make_event(10)],
        results=[make_result(10, selected=False)],
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 0
        assert "TrackClusterEnergy_Z1" in tree


def test_rejects_missing_result(tmp_path):
    with pytest.raises(ValueError, match="Missing TrackingResult"):
        run_golden_tracking_output(
            events=[make_event(10)],
            results=[],
            output_path=tmp_path / "invalid.root",
        )


def test_rejects_unexpected_result(tmp_path):
    with pytest.raises(ValueError, match="no matching event"):
        run_golden_tracking_output(
            events=[make_event(10)],
            results=[make_result(10), make_result(11)],
            output_path=tmp_path / "invalid.root",
        )


def test_rejects_duplicate_event_identity(tmp_path):
    with pytest.raises(ValueError, match="Duplicate TrackingEvent"):
        run_golden_tracking_output(
            events=[make_event(10), make_event(10)],
            results=[make_result(10)],
            output_path=tmp_path / "invalid.root",
        )
