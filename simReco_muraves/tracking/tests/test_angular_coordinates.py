# simReco_muraves/tracking/tests/test_angular_coordinates.py

"""The experiment's angle convention, pinned.

TrackAngularCoordinates is the experiment's own code. These are
characterisation tests: they record what it does now, including edge
cases, so that any change to the convention is deliberate. They do not
claim the edge cases are the preferable choice.

The agreed convention: theta is the elevation above the xy-plane and
is never negative; the sign of dz separates the forward-going branch,
which points into the field of view, from the backward one.
"""

from __future__ import annotations

import math

import pytest

from simReco_muraves.tracking.Evaluate_Angular_Coordinates import (
    TrackAngularCoordinates,
    track_angular_coordinates,
)


# ROSSO working point, as used by the entry points.
X0, X2, X3 = -0.265, 0.262, 1.475


def angles(slope_xy, slope_xz, x_first=X0, x_last=X2):
    return TrackAngularCoordinates(slope_xy, slope_xz, x_first, x_last)


def test_returns_theta_then_phi():
    assert len(angles(0.1, 0.2)) == 2


def test_snake_case_alias_is_the_same_function():
    assert track_angular_coordinates is TrackAngularCoordinates


# ----------------------------------------------------------------------
# Theta
# ----------------------------------------------------------------------

def test_flat_track_has_zero_theta():
    assert angles(0.3, 0.0)[0] == 0.0


def test_unit_xz_slope_gives_45_degrees():
    assert angles(0.0, 1.0)[0] == pytest.approx(45.0)


def test_theta_is_elevation_above_the_xy_plane():
    """theta = atan(|slope_xz| / sqrt(1 + slope_xy^2))."""
    expected = math.degrees(math.atan(1.0 / math.sqrt(2.0)))

    assert angles(1.0, 1.0)[0] == pytest.approx(expected)


@pytest.mark.parametrize("slope_xz", [0.05, 0.4, 1.3])
def test_theta_does_not_depend_on_the_sign_of_the_xz_slope(slope_xz):
    assert angles(0.2, slope_xz)[0] == pytest.approx(angles(0.2, -slope_xz)[0])


@pytest.mark.parametrize("slope_xy", [-1.5, -0.2, 0.0, 0.2, 1.5])
@pytest.mark.parametrize("slope_xz", [-0.8, 0.0, 0.8])
def test_theta_is_never_negative(slope_xy, slope_xz):
    assert angles(slope_xy, slope_xz)[0] >= 0.0


@pytest.mark.parametrize(
    "slope_xy, slope_xz",
    [(0.1, 0.3), (-0.4, -0.2), (0.0, 0.7), (1.2, -0.05)],
)
def test_angles_do_not_depend_on_the_lever_arm(slope_xy, slope_xz):
    """Only the slopes matter, not which two planes are passed.

    The three-plane call uses X_POS[0] and X_POS[2]; the four-plane call
    uses X_POS[0] and X_POS[3]. For a straight track both must agree.
    """
    assert angles(slope_xy, slope_xz, X0, X2) == pytest.approx(
        angles(slope_xy, slope_xz, X0, X3)
    )


# ----------------------------------------------------------------------
# Phi
# ----------------------------------------------------------------------

@pytest.mark.parametrize("slope_xy", [-2.0, -0.3, 0.0, 0.3, 2.0])
def test_forward_tracks_point_into_the_field_of_view(slope_xy):
    """dz > 0 is the downward-going branch; MURAVES sees phi in [90, 270].

    With X0 < X2, a negative xz slope gives dz > 0.
    """
    assert 90.0 < angles(slope_xy, -0.5)[1] < 270.0


def test_forward_track_with_flat_xy_projection_is_at_180():
    assert angles(0.0, -0.5)[1] == pytest.approx(180.0)


def test_backward_tracks_with_positive_xy_slope_are_in_the_first_quadrant():
    assert 0.0 < angles(0.3, 0.5)[1] < 90.0


def test_backward_tracks_with_negative_xy_slope_are_in_the_fourth_quadrant():
    assert 270.0 < angles(-0.3, 0.5)[1] < 360.0


def test_backward_track_with_flat_xy_projection_gives_360():
    """Inherited edge case: 360, not 0, when dz <= 0 and slope_xy == 0.

    Pinned so that a change is noticed, not because 360 is preferable.
    """
    assert angles(0.0, 0.5)[1] == 360.0
