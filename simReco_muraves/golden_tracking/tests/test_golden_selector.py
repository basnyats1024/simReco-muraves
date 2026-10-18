from types import SimpleNamespace

from simReco_muraves.golden_tracking.golden_selector import (
    GoldenRejectCode,
    select_golden_track,
)


def valid_golden_event(**overrides):
    values = {
        "best_track_3p_chi_square_xz": 1.0,
        "best_track_4p_chi_square_xz": 2.0,
        "best_track_3p_xz_index": 0,
        "best_track_3p_xy_index": 1,
        "track_3p_of_4p_index_xz": 2,
        "track_3p_of_4p_index_xy": 3,
        "track_4p_index_xz": 4,
        "track_4p_index_xy": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_golden_gate_rejects_invalid_3p_xz():
    result = select_golden_track(
        valid_golden_event(best_track_3p_chi_square_xz=-1)
    )

    assert result.accepted is False
    assert result.reject_code == GoldenRejectCode.INVALID_3P_XZ


def test_golden_gate_rejects_invalid_4p_xz():
    result = select_golden_track(
        valid_golden_event(best_track_4p_chi_square_xz=-1)
    )

    assert result.accepted is False
    assert result.reject_code == GoldenRejectCode.INVALID_4P_XZ


def test_golden_gate_preserves_all_selected_indices():
    result = select_golden_track(valid_golden_event())

    assert result.accepted is True
    assert result.reject_code == GoldenRejectCode.ACCEPTED
    assert result.best_3p_index_xz == 0
    assert result.best_3p_index_xy == 1
    assert result.parent_3p_index_xz == 2
    assert result.parent_3p_index_xy == 3
    assert result.extension_4p_index_xz == 4
    assert result.extension_4p_index_xy == 5


def test_golden_gate_does_not_add_xy_rejection():
    result = select_golden_track(
        valid_golden_event(
            best_track_3p_xy_index=-1,
            track_3p_of_4p_index_xy=-1,
            track_4p_index_xy=-1,
        )
    )

    assert result.accepted is True
