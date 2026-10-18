from __future__ import annotations

from types import SimpleNamespace

from simReco_muraves.golden_tracking.tracking_result_adapter import build_processing_result
from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.models.tracking_event import TrackingEvent


def cluster(station: int, cluster_id: int, energy: float):
    return ClusterRecord(
        process_id=1,
        event=10,
        station=station,
        cluster_id=cluster_id,
        cluster_multiplicity=1,
        cluster_energy=energy,
        cluster_position=float(cluster_id),
        cluster_size=1,
        cluster_time=100.0,
        cluster_time_spread=0.0,
        strip_ids=[cluster_id],
        strip_energies=[energy],
    )


def make_event():
    return TrackingEvent(
        process_id=1,
        event=10,
        z1=[cluster(8, 0, 10.0)],
        z2=[cluster(9, 0, 11.0)],
        z3=[cluster(10, 0, 12.0)],
        z4=[cluster(11, 0, 13.0)],
        y1=[cluster(0, 0, 20.0)],
        y2=[cluster(1, 0, 21.0)],
        y3=[cluster(2, 0, 22.0)],
        y4=[cluster(3, 0, 23.0)],
    )


def tracks():
    return SimpleNamespace(
        cluster_index_1=[0],
        cluster_index_2=[0],
        cluster_index_3=[0],
        cluster_indices_4=[[0.0]],
        ScatteringAngles=[[1.25]],
    )


def make_result(**overrides):
    values = {
        "process_id": 1,
        "event": 10,
        "tracks_xz": tracks(),
        "tracks_xy": tracks(),
        "best_track_xz_index": 0,
        "best_track_xy_index": 0,
        "track_3p_of_4p_index_xz": 0,
        "track_3p_of_4p_index_xy": 0,
        "track_4p_index_xz": 0,
        "track_4p_index_xy": 0,
        "best_track_3p_chi_square_xz": 1.0,
        "best_track_3p_chi_square_xy": 2.0,
        "best_track_4p_chi_square_xz": 3.0,
        "best_track_4p_chi_square_xy": 4.0,
        "theta_4p": 22.5,
        "phi_4p": 180.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_builds_selected_record_from_tracking_result():
    output = build_processing_result(make_event(), make_result())

    assert output.selected is True
    assert output.record is not None
    assert output.record["ProcessID"] == 1
    assert output.record["EventID"] == 10
    assert output.record["Theta"] == 22.5
    assert output.record["Scattering_xz"] == 1.25
    assert output.record["TrackClusterEnergy_Z1"] == 10.0
    assert output.record["TrackClusterEnergy_Y4"] == 23.0


def test_rejects_invalid_3p_xz():
    output = build_processing_result(
        make_event(),
        make_result(best_track_3p_chi_square_xz=-1.0),
    )

    assert output.selected is False
    assert output.rejection_reason == "invalid_3p_xz"


def test_rejects_invalid_4p_xz():
    output = build_processing_result(
        make_event(),
        make_result(best_track_4p_chi_square_xz=-1.0),
    )

    assert output.selected is False
    assert output.rejection_reason == "invalid_4p_xz"


def test_invalid_xy_is_consistency_error_not_physics_rejection():
    output = build_processing_result(
        make_event(),
        make_result(best_track_xy_index=-1),
    )

    assert output.selected is False
    assert output.rejection_reason is None
    assert output.consistency_error is not None


def test_rejects_event_identity_mismatch_as_consistency_error():
    output = build_processing_result(
        make_event(),
        make_result(event=11),
    )

    assert output.selected is False
    assert output.consistency_error == "event_identity_mismatch"


def test_invalid_scattering_index_is_consistency_error():
    output = build_processing_result(
        make_event(),
        make_result(track_4p_index_xy=1),
    )

    assert output.selected is False
    assert output.consistency_error is not None
