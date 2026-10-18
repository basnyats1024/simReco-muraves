from __future__ import annotations

from simReco_muraves.golden_tracking.event_cluster_output import (
    PlaneClusterInput,
    build_golden_event_cluster_output,
)
from simReco_muraves.golden_tracking.final_output import (
    build_cluster_output_branches,
)
from simReco_muraves.golden_tracking.event_cluster_output import (
    SelectedClusterIndices,
)


def make_plane(
    base: float,
    count: int,
) -> PlaneClusterInput:
    energies = [
        base + 10.0 * index
        for index in range(count)
    ]

    positions = [
        base + float(index)
        for index in range(count)
    ]

    sizes = [
        index + 1
        for index in range(count)
    ]

    texp = [
        100.0 + base + float(index)
        for index in range(count)
    ]

    strip_energies = [
        [
            energies[index] - float(strip_index)
            for strip_index in range(sizes[index])
        ]
        for index in range(count)
    ]

    return PlaneClusterInput(
        energies=energies,
        positions=positions,
        sizes=sizes,
        texp=texp,
        strip_energies=strip_energies,
    )


def build_test_output():
    planes = {
        "z1": make_plane(10.0, 3),
        "z2": make_plane(20.0, 4),
        "z3": make_plane(30.0, 5),
        "z4": make_plane(40.0, 4),
        "y1": make_plane(50.0, 2),
        "y2": make_plane(60.0, 3),
        "y3": make_plane(70.0, 4),
        "y4": make_plane(80.0, 3),
    }

    selected = SelectedClusterIndices(
        z1=2,
        z2=3,
        z3=4,
        z4=3,
        y1=1,
        y2=2,
        y3=3,
        y4=2,
    )

    return build_golden_event_cluster_output(
        selected=selected,
        planes=planes,
    )


def test_builds_expected_number_of_cluster_branches():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    branches_per_plane = 11
    number_of_planes = 8

    assert len(branches) == (
        branches_per_plane * number_of_planes
    )


def test_uses_correct_cluster_multiplicity_spelling():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    assert "ClusterMultiplicity_Z1" in branches
    assert "ClusterMultiplicity_Z4" in branches
    assert "ClusterMultiplicity_Y1" in branches
    assert "ClusterMultiplicity_Y4" in branches

    assert "ClusterMultriplicity_Z1" not in branches
    assert "ClusterMultriplicity_Y1" not in branches


def test_writes_selected_cluster_values():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    assert branches["TrackClusterIndex_Z1"] == 2
    assert branches["TrackClusterEnergy_Z1"] == 30.0
    assert branches["TrackClusterPosition_Z1"] == 12.0
    assert branches["TrackClusterSize_Z1"] == 3
    assert branches["TrackClusterTexp_Z1"] == 112.0

    assert (
        branches["TrackClusterMaxStripEnergy_Z1"]
        == 30.0
    )
    assert (
        branches["TrackClusterMinStripEnergy_Z1"]
        == 28.0
    )


def test_writes_selected_values_for_all_planes():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    expected_indices = {
        "Z1": 2,
        "Z2": 3,
        "Z3": 4,
        "Z4": 3,
        "Y1": 1,
        "Y2": 2,
        "Y3": 3,
        "Y4": 2,
    }

    for suffix, expected_index in expected_indices.items():
        assert (
            branches[f"TrackClusterIndex_{suffix}"]
            == expected_index
        )


def test_writes_correct_multiplicities():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    assert branches["ClusterMultiplicity_Z1"] == 3
    assert branches["ClusterMultiplicity_Z2"] == 4
    assert branches["ClusterMultiplicity_Z3"] == 5
    assert branches["ClusterMultiplicity_Z4"] == 4

    assert branches["ClusterMultiplicity_Y1"] == 2
    assert branches["ClusterMultiplicity_Y2"] == 3
    assert branches["ClusterMultiplicity_Y3"] == 4
    assert branches["ClusterMultiplicity_Y4"] == 3


def test_writes_nontrack_cluster_values():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    assert branches["NonTrackClusterEnergy_Z1"] == [
        10.0,
        20.0,
    ]
    assert branches["NonTrackClusterPosition_Z1"] == [
        10.0,
        11.0,
    ]
    assert branches["NonTrackClusterSize_Z1"] == [
        1,
        2,
    ]


def test_output_vector_branches_are_independent_lists():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    energies = branches["NonTrackClusterEnergy_Z1"]
    positions = branches["NonTrackClusterPosition_Z1"]
    sizes = branches["NonTrackClusterSize_Z1"]

    assert isinstance(energies, list)
    assert isinstance(positions, list)
    assert isinstance(sizes, list)

    energies.append(999.0)
    positions.append(999.0)
    sizes.append(999)

    assert branches["NonTrackClusterEnergy_Z2"] != energies
    assert branches["NonTrackClusterPosition_Z2"] != positions
    assert branches["NonTrackClusterSize_Z2"] != sizes


def test_all_expected_plane_suffixes_exist():
    branches = build_cluster_output_branches(
        build_test_output()
    )

    expected_suffixes = (
        "Z1",
        "Z2",
        "Z3",
        "Z4",
        "Y1",
        "Y2",
        "Y3",
        "Y4",
    )

    prefixes = (
        "TrackClusterIndex",
        "TrackClusterEnergy",
        "TrackClusterPosition",
        "TrackClusterSize",
        "TrackClusterTexp",
        "TrackClusterMaxStripEnergy",
        "TrackClusterMinStripEnergy",
        "ClusterMultiplicity",
        "NonTrackClusterEnergy",
        "NonTrackClusterPosition",
        "NonTrackClusterSize",
    )

    for suffix in expected_suffixes:
        for prefix in prefixes:
            assert f"{prefix}_{suffix}" in branches
