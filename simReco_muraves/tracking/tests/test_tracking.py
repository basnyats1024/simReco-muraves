# simReco_muraves/tracking/tests/test_tracking.py

"""The straight-line fit and candidate building in MakeTracks.

Tracking.py is the experiment's own code, reused unmodified. These are
characterisation tests on synthetic straight tracks: they pin what the
code does, including two behaviours recorded in the README as
observations rather than fixes -- the Texp gate and the unused plane-4
offset -- so that changing either is a deliberate act.

MakeTracks fits ``position + Z_add = intercept + slope * X_pos``, so a
track with a known intercept and slope can be built exactly.
"""

from __future__ import annotations

import pytest

from simReco_muraves.tracking.Tracking import (
    MakeTracks,
    _linear_fit_with_known_sigma,
)


# ROSSO working point and cuts, as used by the entry points.
X_POS = [-0.265, 0.0, 0.262, 1.475]
Z_ADD = [0.292, 0.251, 0.207, 0.0]
SIGMA = 0.004
PROXIMITY = 0.020


def measured_positions(intercept, slope, z_add=Z_ADD):
    """Cluster positions of the straight track z = intercept + slope * x."""
    return [intercept + slope * x - offset for x, offset in zip(X_POS, z_add)]


def make_tracks(c1, c2, c3, c4, *, texp=(1.0, 1.0, 1.0), texp4=None, z_add=Z_ADD):
    texp1, texp2, texp3 = texp
    texp4 = [1.0] * len(c4) if texp4 is None else texp4

    return MakeTracks(
        c1, c2, c3, c4,
        [30.0] * len(c1), [30.0] * len(c2), [30.0] * len(c3), [30.0] * len(c4),
        [texp1] * len(c1), [texp2] * len(c2), [texp3] * len(c3), texp4,
        PROXIMITY, X_POS, z_add, SIGMA,
    )


def one_track(intercept=0.05, slope=0.1, **kwargs):
    p = measured_positions(intercept, slope)
    return make_tracks([p[0]], [p[1]], [p[2]], [p[3]], **kwargs), p


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------

def test_fit_recovers_a_straight_line():
    intercept, slope, chi_square, _, _ = _linear_fit_with_known_sigma(
        [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 0.1
    )

    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)
    assert chi_square == pytest.approx(0.0, abs=1e-12)


def test_fit_chi_square_scales_with_inverse_sigma_squared():
    x, y = [0.0, 1.0, 2.0], [0.0, 0.1, 0.0]

    chi_small = _linear_fit_with_known_sigma(x, y, 0.1)[2]
    chi_large = _linear_fit_with_known_sigma(x, y, 0.2)[2]

    assert chi_small == pytest.approx(4.0 * chi_large)


def test_fit_needs_at_least_two_points():
    assert _linear_fit_with_known_sigma([1.0], [2.0], 0.1) == (0.0,) * 5


def test_fit_of_coincident_x_gives_zero_slope():
    assert _linear_fit_with_known_sigma([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 0.1)[1] == 0.0


# ----------------------------------------------------------------------
# Three-plane candidates
# ----------------------------------------------------------------------

def test_straight_track_is_found_and_fitted():
    tracks, _ = one_track(intercept=0.05, slope=0.1)

    assert len(tracks.slopes_3p) == 1
    assert tracks.slopes_3p[0] == pytest.approx(0.1)
    assert tracks.intercepts_3p[0] == pytest.approx(0.05)
    assert tracks.chiSquares_3p[0] == pytest.approx(0.0, abs=1e-9)
    assert tracks.BestChi_index == 0


def test_plane_2_outside_the_proximity_cut_gives_no_track():
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks([p[0]], [p[1] + 0.05], [p[2]], [p[3]])

    assert tracks.slopes_3p == []
    assert tracks.BestChi_index == -1


def test_plane_2_inside_the_cut_gives_a_track_with_nonzero_chi_square():
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks([p[0]], [p[1] + 0.01], [p[2]], [p[3]])

    assert len(tracks.slopes_3p) == 1
    assert tracks.chiSquares_3p[0] > 0.0


def test_best_chi_square_candidate_is_selected():
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks([p[0]], [p[1] + 0.012, p[1]], [p[2]], [p[3]])

    assert len(tracks.slopes_3p) == 2
    assert tracks.BestChi_index == 1
    assert tracks.cluster_index_2[tracks.BestChi_index] == 1


def test_texp_gate_requires_positive_time_on_plane_1():
    tracks, _ = one_track(texp=(0.0, 1.0, 1.0))

    assert len(tracks.slopes_3p) == 1
    assert tracks.BestChi_index == -1


def test_texp_gate_only_tests_the_sign_on_plane_1():
    """Pinned quirk: planes 2 and 3 are truthiness tests, not > 0.

    A negative Texp there passes the gate. Simulation is unaffected,
    since simulated times are positive. See the README.
    """
    tracks, _ = one_track(texp=(1.0, -1.0, -1.0))

    assert tracks.BestChi_index == 0


# ----------------------------------------------------------------------
# Four-plane extension
# ----------------------------------------------------------------------

def test_straight_track_is_extended_to_plane_4():
    tracks, _ = one_track(intercept=0.05, slope=0.1)

    assert tracks.Track_3p_to_4p_index == [0]
    assert tracks.Plane4th_isIntercepted == [1]
    assert tracks.slope_4p[0][0] == pytest.approx(0.1)
    assert tracks.chiSquares_4p[0][0] == pytest.approx(0.0, abs=1e-9)
    assert tracks.ScatteringAngles[0][0] == pytest.approx(0.0, abs=1e-4)


def test_plane_4_candidates_are_sorted_by_distance_without_a_cut():
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks(
        [p[0]], [p[1]], [p[2]],
        [p[3] + 0.08, p[3] + 0.001, p[3] - 0.03],
    )

    assert tracks.cluster_indices_4[0] == [1.0, 2.0, 0.0]
    assert [abs(d) for d in tracks.displacement_p4[0]] == sorted(
        abs(d) for d in tracks.displacement_p4[0]
    )


@pytest.mark.parametrize("shift, sign", [(0.02, 1.0), (-0.02, -1.0)])
def test_scattering_angle_takes_the_sign_of_the_displacement(shift, sign):
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks([p[0]], [p[1]], [p[2]], [p[3] + shift])

    assert tracks.ScatteringAngles[0][0] * sign > 0.0


def test_extrapolation_outside_plane_4_is_not_extended():
    tracks, _ = one_track(intercept=0.05, slope=0.4)

    assert tracks.Plane4th_isIntercepted == [0]
    assert tracks.Track_3p_to_4p_index == []
    assert tracks.chiSquares_4p == []


def test_empty_plane_4_records_the_intercept_but_no_candidates():
    """Pinned: a nested row is appended, but no Track_3p_to_4p_index entry.

    The two stay in step only because plane-4 clusters are a per-event
    property. See the README.
    """
    p = measured_positions(0.05, 0.1)
    tracks = make_tracks([p[0]], [p[1]], [p[2]], [])

    assert tracks.Plane4th_isIntercepted == [1]
    assert tracks.Track_3p_to_4p_index == []
    assert tracks.chiSquares_4p == [[]]


def test_plane_4_offset_is_not_applied():
    """Pinned quirk: Z_add[3] never reaches plane 4. See the README."""
    p = measured_positions(0.05, 0.1)
    shifted = list(Z_ADD)
    shifted[3] = 0.1

    baseline = make_tracks([p[0]], [p[1]], [p[2]], [p[3]])
    with_offset = make_tracks([p[0]], [p[1]], [p[2]], [p[3]], z_add=shifted)

    assert with_offset.chiSquares_4p == baseline.chiSquares_4p


def test_no_clusters_gives_no_tracks():
    tracks = make_tracks([], [], [], [])

    assert tracks.slopes_3p == []
    assert tracks.BestChi_index == -1
