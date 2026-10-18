from __future__ import annotations

import pytest

from simReco_muraves.golden_tracking.event_cluster_output import (
    PlaneClusterInput,
    build_golden_event_cluster_output,
)
from simReco_muraves.golden_tracking.final_output import (
    GoldenEventIdentity,
    build_final_golden_branches,
)
from simReco_muraves.golden_tracking.golden_root_schema import (
    build_golden_root_schema,
    validate_record_against_golden_schema,
)
from simReco_muraves.golden_tracking.event_cluster_output import SelectedClusterIndices


def make_plane(base: float, count: int) -> PlaneClusterInput:
    energies = [base + 10.0 * index for index in range(count)]
    positions = [base + float(index) for index in range(count)]
    sizes = [index + 1 for index in range(count)]
    texp = [100.0 + base + float(index) for index in range(count)]
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


def build_record():
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

    cluster_output = build_golden_event_cluster_output(
        selected=selected,
        planes=planes,
    )

    return build_final_golden_branches(
        identity=GoldenEventIdentity(
            process_id=1,
            event_id=10,
        ),
        track_fields={
            "Theta": 22.5,
            "Phi": 180.0,
            "Scattering_xz": 1.0,
            "Scattering_xy": -1.0,
            "ChiSquare_xz": 3.0,
            "ChiSquare_xy": 4.0,
            "ChiSquare_3p_xz": 1.0,
            "ChiSquare_3p_xy": 2.0,
        },
        selected_index_fields={
            "BestTrack3pIndex_xz": 1,
            "BestTrack3pIndex_xy": 0,
            "Track3pOf4pRow_xz": 1,
            "Track3pOf4pRow_xy": 0,
            "Track4pInnerIndex_xz": 0,
            "Track4pInnerIndex_xy": 0,
        },
        cluster_output=cluster_output,
    )


def test_schema_has_expected_branch_count():
    schema = build_golden_root_schema()

    identity_branches = 2
    track_branches = 8
    selected_index_branches = 6
    flag_branches = 2
    per_plane_branches = 11
    number_of_planes = 8

    assert len(schema) == (
        identity_branches
        + track_branches
        + selected_index_branches
        + flag_branches
        + per_plane_branches * number_of_planes
    )


def test_schema_uses_correct_multiplicity_spelling():
    schema = build_golden_root_schema()

    assert "ClusterMultiplicity_Z1" in schema
    assert "ClusterMultiplicity_Y4" in schema
    assert "ClusterMultriplicity_Z1" not in schema
    assert "ClusterMultriplicity_Y4" not in schema


def test_schema_defines_jagged_nontrack_branches():
    schema = build_golden_root_schema()

    assert schema["NonTrackClusterEnergy_Z1"] == "var * float64"
    assert schema["NonTrackClusterPosition_Z1"] == "var * float64"
    assert schema["NonTrackClusterSize_Z1"] == "var * int64"


def test_schema_defines_selected_cluster_branches():
    schema = build_golden_root_schema()

    assert schema["TrackClusterIndex_Z1"] == "int64"
    assert schema["TrackClusterEnergy_Z1"] == "float64"
    assert schema["TrackClusterPosition_Z1"] == "float64"
    assert schema["TrackClusterSize_Z1"] == "int64"
    assert schema["TrackClusterTexp_Z1"] == "float64"
    assert schema["TrackClusterMaxStripEnergy_Z1"] == "float64"
    assert schema["TrackClusterMinStripEnergy_Z1"] == "float64"


def test_final_record_matches_schema_exactly():
    record = build_record()
    schema = build_golden_root_schema()

    assert tuple(record.keys()) == tuple(schema.keys())
    validate_record_against_golden_schema(record)


def test_rejects_missing_branch():
    record = build_record()
    del record["Theta"]

    with pytest.raises(
        ValueError,
        match="missing branches: Theta",
    ):
        validate_record_against_golden_schema(record)


def test_rejects_unexpected_branch():
    record = build_record()
    record["Unexpected"] = 1

    with pytest.raises(
        ValueError,
        match="unexpected branches: Unexpected",
    ):
        validate_record_against_golden_schema(record)


def test_schema_supports_zero_entry_root_output():
    schema = build_golden_root_schema()

    assert schema
    assert all(isinstance(value, str) for value in schema.values())
