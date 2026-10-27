# simReco_muraves/tests/test_nested_access.py

"""Bounds-checked access to doubly-nested four-plane quantities.

Both golden routes and the reconstruction adapter read four-plane
values through these helpers, so the guarantees are shared: an
unreachable index never raises unexpectedly, and -1 is never read as
Python's "last element".
"""

from __future__ import annotations

import numpy as np
import pytest

from simReco_muraves.nested_access import (
    MISSING_VALUE,
    nested_value,
    nested_value_or_missing,
    nested_value_strict,
)


VALUES = [[1.0, 2.0], [], [3.0]]


def test_reads_a_nested_value():
    assert nested_value(VALUES, 0, 1) == 2.0
    assert nested_value(VALUES, 2, 0) == 3.0


@pytest.mark.parametrize(
    "parent, inner",
    [(3, 0), (0, 2), (1, 0), (-1, 0), (0, -1)],
)
def test_unreachable_index_gives_none(parent, inner):
    assert nested_value(VALUES, parent, inner) is None


def test_minus_one_is_never_the_last_element():
    """-1 means "no candidate" throughout the chain.

    Read as Python's last element, an invalid index would turn into a
    plausible-looking selection instead of a rejection.
    """
    assert nested_value([[5.0]], -1, -1) is None
    assert nested_value([[5.0]], 0, -1) is None


@pytest.mark.parametrize("index", [True, 1.0, "1", None])
def test_non_integer_indices_are_refused(index):
    assert nested_value(VALUES, index, 0) is None
    assert nested_value(VALUES, 0, index) is None


def test_numpy_integers_are_accepted():
    assert nested_value(VALUES, np.int64(0), np.int32(1)) == 2.0


def test_empty_values():
    assert nested_value([], 0, 0) is None


def test_or_missing_uses_the_chain_wide_sentinel():
    assert MISSING_VALUE == -1.0
    assert nested_value_or_missing(VALUES, 9, 9) == MISSING_VALUE


def test_or_missing_accepts_a_custom_sentinel():
    assert nested_value_or_missing(VALUES, 9, 9, missing=-99.0) == -99.0


def test_or_missing_returns_floats():
    value = nested_value_or_missing([[3]], 0, 0)

    assert value == 3.0
    assert isinstance(value, float)


def test_strict_returns_the_value():
    assert nested_value_strict(VALUES, 0, 0, "Scattering_xz") == 1.0


def test_strict_raises_naming_the_field():
    with pytest.raises(IndexError, match="Scattering_xz"):
        nested_value_strict(VALUES, 5, 0, "Scattering_xz")
