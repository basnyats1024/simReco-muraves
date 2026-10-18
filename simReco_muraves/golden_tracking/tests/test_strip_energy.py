from __future__ import annotations

import math

import pytest

from simReco_muraves.golden_tracking.event_cluster_output import summarize_strip_energies


def test_extracts_first_as_maximum_and_last_as_minimum():
    result = summarize_strip_energies(
        strip_energies=[12.0, 8.0, 3.0],
        cluster_size=3,
    )

    assert result.maximum == 12.0
    assert result.minimum == 3.0
    assert result.count == 3
    assert result.is_descending is True


def test_single_strip_cluster():
    result = summarize_strip_energies(
        strip_energies=[7.5],
        cluster_size=1,
    )

    assert result.maximum == 7.5
    assert result.minimum == 7.5
    assert result.count == 1
    assert result.is_descending is True


def test_equal_adjacent_energies_are_allowed():
    result = summarize_strip_energies(
        strip_energies=[10.0, 10.0, 4.0, 4.0],
        cluster_size=4,
    )

    assert result.maximum == 10.0
    assert result.minimum == 4.0
    assert result.is_descending is True


def test_input_collection_is_not_modified():
    energies = [12.0, 8.0, 3.0]

    summarize_strip_energies(
        strip_energies=energies,
        cluster_size=3,
    )

    assert energies == [12.0, 8.0, 3.0]


def test_rejects_empty_strip_energy_list():
    with pytest.raises(ValueError, match="positive"):
        summarize_strip_energies(
            strip_energies=[],
            cluster_size=0,
        )


@pytest.mark.parametrize(
    "cluster_size",
    [
        -1,
        -10,
    ],
)
def test_rejects_negative_cluster_size(cluster_size):
    with pytest.raises(ValueError, match="positive"):
        summarize_strip_energies(
            strip_energies=[10.0],
            cluster_size=cluster_size,
        )


@pytest.mark.parametrize(
    "cluster_size",
    [
        True,
        False,
        2.0,
        "2",
        None,
    ],
)
def test_rejects_noninteger_cluster_size(cluster_size):
    with pytest.raises(TypeError, match="integer"):
        summarize_strip_energies(
            strip_energies=[10.0, 5.0],
            cluster_size=cluster_size,
        )


@pytest.mark.parametrize(
    ("energies", "cluster_size"),
    [
        ([10.0, 5.0], 1),
        ([10.0], 2),
        ([], 1),
    ],
)
def test_rejects_cluster_size_mismatch(
    energies,
    cluster_size,
):
    with pytest.raises(ValueError, match="match cluster_size"):
        summarize_strip_energies(
            strip_energies=energies,
            cluster_size=cluster_size,
        )


@pytest.mark.parametrize(
    "energies",
    [
        [10.0, math.nan, 2.0],
        [10.0, math.inf, 2.0],
        [10.0, -math.inf, 2.0],
    ],
)
def test_rejects_nonfinite_strip_energies(energies):
    with pytest.raises(ValueError, match="finite"):
        summarize_strip_energies(
            strip_energies=energies,
            cluster_size=3,
        )


@pytest.mark.parametrize(
    "energies",
    [
        [5.0, 10.0],
        [10.0, 3.0, 4.0],
        [1.0, 2.0, 3.0],
    ],
)
def test_rejects_unsorted_strip_energies(energies):
    with pytest.raises(ValueError, match="highest to lowest"):
        summarize_strip_energies(
            strip_energies=energies,
            cluster_size=len(energies),
        )


def test_returns_independent_numeric_values():
    energies = [12, 8, 3]

    result = summarize_strip_energies(
        strip_energies=energies,
        cluster_size=3,
    )

    assert result.maximum == 12.0
    assert result.minimum == 3.0
    assert isinstance(result.maximum, float)
    assert isinstance(result.minimum, float)
