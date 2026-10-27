# simReco_muraves/tracking/tests/test_reconstruct_event.py

"""reconstruct_event: plane-4 selection, and whole tracks end to end.

The end-to-end tests push a synthetic straight track with known slopes
through the full per-event reconstruction and compare the reconstructed
angles with the ones the slopes imply. That makes them a regression
test for the Phase II geometry bug, where tracking ran on plane indices
instead of plane positions and the three-plane theta came out near
10.9 degrees instead of 32.4.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from simReco_muraves.config.reco_config import get_reco_config
from simReco_muraves.models.cluster_record import ClusterRecord
from simReco_muraves.models.tracking_event import TrackingEvent
from simReco_muraves.tracking import reconstruction_adapter as ra
from simReco_muraves.tracking.Evaluate_Angular_Coordinates import (
    TrackAngularCoordinates,
)
from simReco_muraves.tracking.event_builder_adapter import (
    build_tracking_events,
)


X_POS = [-0.265, 0.0, 0.262, 1.475]
Z_ADD = [0.292, 0.251, 0.207, 0.0]
Y_ADD = [0.0, 0.0, 0.0, 0.0]

SLOPE_XZ = 0.12
SLOPE_XY = 0.08


# ----------------------------------------------------------------------
# Plane-4 selection helpers
# ----------------------------------------------------------------------

def test_texp_flag_values():
    assert ra.TEXP_NULL_NOT_PROMOTED == -1
    assert ra.TEXP_NULL_TRUE == 1
    assert ra.TEXP_NULL_FALSE == 0


def test_best_3p_index_is_minus_one_without_tracks():
    tracks = SimpleNamespace(BestChi_index=3, BestEnergy_index=3)

    assert ra._select_best_3p_index(tracks, 0) == -1


@pytest.mark.parametrize("best_chi, best_energy", [(2, 2), (2, 5), (-1, 4)])
def test_best_3p_index_is_the_best_chi_square(best_chi, best_energy):
    """Ported verbatim; both branches currently give BestChi_index."""
    tracks = SimpleNamespace(BestChi_index=best_chi, BestEnergy_index=best_energy)

    assert ra._select_best_3p_index(tracks, 4) == best_chi


def test_texp_table_follows_the_candidate_layout():
    plane4 = [SimpleNamespace(cluster_time=t) for t in (5.0, -1.0, 7.0)]

    assert ra._build_plane4_texp_table([[2.0, 0.0], [1.0]], plane4) == [
        [7.0, 5.0],
        [-1.0],
    ]


def test_texp_table_marks_out_of_range_candidates():
    plane4 = [SimpleNamespace(cluster_time=5.0)]

    assert ra._build_plane4_texp_table([[0.0, 3.0]], plane4) == [[5.0, -1.0]]


def test_candidate_is_the_first_with_a_positive_texp():
    assert ra._select_plane4_candidate([[-1.0, 0.0, 4.0, 6.0]], 0) == (
        2,
        ra.TEXP_NULL_FALSE,
    )


def test_candidate_falls_back_to_the_first_without_a_valid_texp():
    assert ra._select_plane4_candidate([[-1.0, 0.0]], 0) == (
        0,
        ra.TEXP_NULL_TRUE,
    )


@pytest.mark.parametrize("row", [-1, 1, 5])
def test_candidate_falls_back_for_an_unreachable_row(row):
    assert ra._select_plane4_candidate([[4.0]], row) == (0, ra.TEXP_NULL_TRUE)


# ----------------------------------------------------------------------
# Whole tracks
# ----------------------------------------------------------------------

def line(intercept, slope, offsets, stations, texps):
    return [
        ClusterRecord(
            process_id=1,
            event=10,
            station=station,
            cluster_id=0,
            cluster_multiplicity=1,
            cluster_energy=30.0,
            cluster_position=intercept + slope * x - offset,
            cluster_size=1,
            cluster_time=texp,
            cluster_time_spread=0.0,
            strip_ids=[10],
            strip_energies=[30.0],
        )
        for x, offset, station, texp in zip(X_POS, offsets, stations, texps)
    ]


def straight_event(texp_plane4=1.0):
    texps = (1.0, 1.0, 1.0, texp_plane4)
    clusters = (
        line(0.02, SLOPE_XZ, Z_ADD, (8, 9, 10, 11), texps)
        + line(-0.03, SLOPE_XY, Y_ADD, (0, 1, 2, 3), texps)
    )
    return build_tracking_events(clusters)[0]


def reconstruct(event, x_pos=X_POS):
    return ra.reconstruct_event(
        event,
        x_pos=x_pos,
        z_add=Z_ADD,
        y_add=Y_ADD,
        sigma_z=0.004,
        sigma_y=0.0035,
        proximity_cut_xz=0.020,
        proximity_cut_xy=0.0175,
        tracking_cfg=get_reco_config()["tracking"],
    )


def expected_angles():
    return TrackAngularCoordinates(SLOPE_XY, SLOPE_XZ, X_POS[0], X_POS[3])


def test_identity_is_carried_through():
    result = reconstruct(straight_event())

    assert (result.process_id, result.event) == (1, 10)


def test_straight_track_is_promoted_to_four_planes():
    result = reconstruct(straight_event())

    assert result.best_track_xz_index == 0
    assert result.best_track_xy_index == 0
    assert result.track_3p_of_4p_index_xz == 0
    assert result.track_4p_index_xz == 0
    assert result.best_track_4p_is_texp_null_xz == ra.TEXP_NULL_FALSE
    assert result.best_track_4p_is_texp_null_xy == ra.TEXP_NULL_FALSE
    assert result.best_track_3p_chi_square_xz == pytest.approx(0.0, abs=1e-9)
    assert result.best_track_4p_chi_square_xz == pytest.approx(0.0, abs=1e-9)
    assert result.best_track_4p_chi_square_xy == pytest.approx(0.0, abs=1e-9)


def test_four_plane_angles_match_the_generating_slopes():
    """Regression test for the Phase II plane-index geometry bug."""
    result = reconstruct(straight_event())

    theta, phi = expected_angles()

    assert result.theta_4p == pytest.approx(theta)
    assert result.phi_4p == pytest.approx(phi)


def test_three_plane_angles_match_the_generating_slopes():
    result = reconstruct(straight_event())

    theta, phi = expected_angles()

    assert result.theta_3p == pytest.approx(theta)
    assert result.phi_3p == pytest.approx(phi)


def test_plane_indices_instead_of_positions_would_be_caught():
    """Shows the regression test above is sensitive to the old bug."""
    result = reconstruct(straight_event(), x_pos=[0.0, 1.0, 2.0, 3.0])

    assert result.theta_4p != pytest.approx(expected_angles()[0], abs=0.5)


def test_without_a_valid_plane_4_time_the_nearest_candidate_is_kept():
    result = reconstruct(straight_event(texp_plane4=-1.0))

    assert result.track_4p_index_xz == 0
    assert result.best_track_4p_is_texp_null_xz == ra.TEXP_NULL_TRUE
    assert result.best_track_4p_is_texp_null_xy == ra.TEXP_NULL_TRUE
    assert result.best_track_4p_chi_square_xz >= 0.0


def test_nearest_candidate_with_a_valid_time_wins():
    event = straight_event()
    on_line = event.z4[0]
    decoy = ClusterRecord(**{
        **vars(on_line),
        "cluster_position": on_line.cluster_position + 0.001,
        "cluster_time": -1.0,
    })
    event.z4 = [decoy, ClusterRecord(**{
        **vars(on_line),
        "cluster_position": on_line.cluster_position + 0.01,
    })]

    result = reconstruct(event)

    parent = result.track_3p_of_4p_index_xz
    inner = result.track_4p_index_xz

    assert inner == 1
    assert result.best_track_4p_is_texp_null_xz == ra.TEXP_NULL_FALSE
    assert result.tracks_xz.cluster_indices_4[parent][inner] == 1.0


def test_missing_plane_4_in_one_view_blocks_promotion_in_both():
    event = straight_event()
    event.z4 = []

    result = reconstruct(event)

    assert result.track_3p_of_4p_index_xz == -1
    assert result.track_3p_of_4p_index_xy == -1
    assert result.best_track_4p_is_texp_null_xz == ra.TEXP_NULL_NOT_PROMOTED
    assert result.best_track_4p_is_texp_null_xy == ra.TEXP_NULL_NOT_PROMOTED
    assert result.best_track_4p_chi_square_xz == -1.0
    assert result.theta_4p == -1.0


def test_empty_event_reconstructs_to_nothing():
    result = reconstruct(TrackingEvent(process_id=1, event=10))

    assert result.best_track_xz_index == -1
    assert result.best_track_3p_chi_square_xz == -1.0
    assert result.theta_3p == -1.0
    assert result.best_track_4p_is_texp_null_xz == ra.TEXP_NULL_NOT_PROMOTED
