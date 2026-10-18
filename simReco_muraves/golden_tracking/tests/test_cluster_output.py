from __future__ import annotations

import math

import pytest

from simReco_muraves.golden_tracking.event_cluster_output import build_plane_cluster_output


def valid_plane_state() -> dict:
    return {
        "energies": [10.0, 20.0, 30.0],
        "positions": [1.0, 2.0, 3.0],
        "sizes": [1, 2, 3],
        "texp": [101.0, 102.0, 103.0],
        "strip_energies": [
            [10.0],
            [12.0, 8.0],
            [15.0, 10.0, 5.0],
        ],
        "selected_index": 1,
    }


def test_builds_selected_plane_cluster_output():
    result = build_plane_cluster_output(
        **valid_plane_state()
    )

    assert result.selected.index == 1
    assert result.selected.energy == 20.0
    assert result.selected.position == 2.0
    assert result.selected.size == 2
    assert result.selected.texp == 102.0
    assert result.selected.maximum_strip_energy == 12.0
    assert result.selected.minimum_strip_energy == 8.0


def test_builds_nontrack_plane_cluster_output():
    result = build_plane_cluster_output(
        **valid_plane_state()
    )

    assert result.nontrack.energies == [10.0, 30.0]
    assert result.nontrack.positions == [1.0, 3.0]
    assert result.nontrack.sizes == [1, 3]


def test_records_cluster_multiplicity():
    result = build_plane_cluster_output(
        **valid_plane_state()
    )

    assert result.multiplicity == 3


def test_selected_cluster_is_excluded_by_index():
    state = valid_plane_state()
    state["energies"] = [10.0, 10.0, 30.0]
    state["positions"] = [1.0, 1.0, 3.0]
    state["sizes"] = [1, 1, 3]
    state["texp"] = [101.0, 101.0, 103.0]
    state["strip_energies"] = [
        [10.0],
        [10.0],
        [15.0, 10.0, 5.0],
    ]
    state["selected_index"] = 0

    result = build_plane_cluster_output(**state)

    assert result.selected.index == 0
    assert result.nontrack.energies == [10.0, 30.0]
    assert result.nontrack.positions == [1.0, 3.0]
    assert result.nontrack.sizes == [1, 3]


@pytest.mark.parametrize(
    "field",
    [
        "positions",
        "sizes",
        "texp",
        "strip_energies",
    ],
)
def test_rejects_mismatched_collection_lengths(field):
    state = valid_plane_state()
    state[field] = state[field][:-1]

    with pytest.raises(ValueError, match="same length"):
        build_plane_cluster_output(**state)


@pytest.mark.parametrize(
    "selected_index",
    [
        -1,
        3,
    ],
)
def test_rejects_out_of_range_selected_index(selected_index):
    state = valid_plane_state()
    state["selected_index"] = selected_index

    with pytest.raises(IndexError):
        build_plane_cluster_output(**state)


@pytest.mark.parametrize(
    "selected_index",
    [
        True,
        False,
        1.0,
        "1",
        None,
    ],
)
def test_rejects_noninteger_selected_index(selected_index):
    state = valid_plane_state()
    state["selected_index"] = selected_index

    with pytest.raises(TypeError, match="integer"):
        build_plane_cluster_output(**state)


def test_rejects_selected_strip_count_mismatch():
    state = valid_plane_state()
    state["strip_energies"][1] = [12.0]

    with pytest.raises(ValueError, match="match cluster_size"):
        build_plane_cluster_output(**state)


def test_rejects_unsorted_selected_strip_energies():
    state = valid_plane_state()
    state["strip_energies"][1] = [8.0, 12.0]

    with pytest.raises(ValueError, match="highest to lowest"):
        build_plane_cluster_output(**state)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_nonfinite_selected_strip_energy(value):
    state = valid_plane_state()
    state["strip_energies"][1] = [12.0, value]

    with pytest.raises(ValueError, match="finite"):
        build_plane_cluster_output(**state)


def test_input_collections_are_not_modified():
    state = valid_plane_state()

    original_energies = list(state["energies"])
    original_positions = list(state["positions"])
    original_sizes = list(state["sizes"])
    original_texp = list(state["texp"])
    original_strip_energies = [
        list(values)
        for values in state["strip_energies"]
    ]

    build_plane_cluster_output(**state)

    assert state["energies"] == original_energies
    assert state["positions"] == original_positions
    assert state["sizes"] == original_sizes
    assert state["texp"] == original_texp
    assert state["strip_energies"] == original_strip_energies


def test_result_nontrack_lists_are_independent():
    state = valid_plane_state()

    result = build_plane_cluster_output(**state)

    assert result.nontrack.energies is not state["energies"]
    assert result.nontrack.positions is not state["positions"]
    assert result.nontrack.sizes is not state["sizes"]
