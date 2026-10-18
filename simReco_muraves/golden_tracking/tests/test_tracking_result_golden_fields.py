from __future__ import annotations

from dataclasses import fields

from simReco_muraves.models.tracking_results import TrackingResult


def test_tracking_result_exposes_golden_selection_state():
    names = {field.name for field in fields(TrackingResult)}

    expected = {
        "track_3p_of_4p_index_xz",
        "track_3p_of_4p_index_xy",
        "track_4p_index_xz",
        "track_4p_index_xy",
        "best_track_3p_chi_square_xz",
        "best_track_3p_chi_square_xy",
        "best_track_4p_chi_square_xz",
        "best_track_4p_chi_square_xy",
    }

    assert expected <= names
