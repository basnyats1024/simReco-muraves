from types import SimpleNamespace

from simReco_muraves.golden_tracking.final_output import build_golden_track_record
from simReco_muraves.golden_tracking.golden_selector import GoldenSelection


def test_rejected_selection_returns_none():
    event = SimpleNamespace()

    selection = GoldenSelection(accepted=False)

    assert build_golden_track_record(event, selection) is None


def test_builds_selected_golden_track_record():
    event = SimpleNamespace(
        theta_4p=22.5,
        phi_4p=180.0,
        scattering_angle_xz=[
            [0.10, 0.20],
            [0.30, 0.40],
        ],
        scattering_angle_xy=[
            [0.50, 0.60],
            [0.70, 0.80],
        ],
        chi_square_4p_xz=[
            [1.1, 1.2],
            [1.3, 1.4],
        ],
        chi_square_4p_xy=[
            [2.1, 2.2],
            [2.3, 2.4],
        ],
        chi_square_3p_xz=[3.1, 3.2],
        chi_square_3p_xy=[4.1, 4.2],
    )

    selection = GoldenSelection(
        accepted=True,
        best_3p_index_xz=1,
        best_3p_index_xy=0,
        parent_3p_index_xz=0,
        parent_3p_index_xy=1,
        extension_4p_index_xz=1,
        extension_4p_index_xy=0,
    )

    record = build_golden_track_record(event, selection)

    assert record is not None
    assert record.theta == 22.5
    assert record.phi == 180.0
    assert record.scattering_xz == 0.20
    assert record.scattering_xy == 0.70
    assert record.chi_square_4p_xz == 1.2
    assert record.chi_square_4p_xy == 2.3
    assert record.chi_square_3p_xz == 3.2
    assert record.chi_square_3p_xy == 4.1
