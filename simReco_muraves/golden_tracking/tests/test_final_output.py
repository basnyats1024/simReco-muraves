from __future__ import annotations

import math

import pytest

from simReco_muraves.golden_tracking.event_cluster_output import (
    PlaneClusterInput,
    build_golden_event_cluster_output,
)
from simReco_muraves.golden_tracking.final_output import (
    REQUIRED_INDEX_FIELDS,
    REQUIRED_TRACK_FIELDS,
    GoldenEventIdentity,
    build_final_golden_branches,
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


def build_test_cluster_output():
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


def valid_track_fields() -> dict[str, float]:
    return {
        "Theta": 22.5,
        "Phi": 180.0,
        "Scattering_xz": 1.25,
        "Scattering_xy": -0.75,
        "ChiSquare_xz": 3.5,
        "ChiSquare_xy": 4.5,
        "ChiSquare_3p_xz": 1.5,
        "ChiSquare_3p_xy": 2.5,
    }


def valid_index_fields() -> dict[str, int]:
    return {
        "BestTrack3pIndex_xz": 3,
        "BestTrack3pIndex_xy": 2,
        "Track3pOf4pRow_xz": 1,
        "Track3pOf4pRow_xy": 0,
        "Track4pInnerIndex_xz": 0,
        "Track4pInnerIndex_xy": 0,
    }


def build_test_branches():
    return build_final_golden_branches(
        identity=GoldenEventIdentity(
            process_id=12,
            event_id=345,
        ),
        track_fields=valid_track_fields(),
        selected_index_fields=valid_index_fields(),
        cluster_output=build_test_cluster_output(),
    )


def test_builds_event_identity_branches():
    branches = build_test_branches()

    assert branches["ProcessID"] == 12
    assert branches["EventID"] == 345


def test_builds_all_track_branches():
    branches = build_test_branches()
    expected = valid_track_fields()

    for field_name in REQUIRED_TRACK_FIELDS:
        assert branches[field_name] == expected[field_name]
        assert isinstance(branches[field_name], float)


def test_builds_all_selected_index_branches():
    branches = build_test_branches()
    expected = valid_index_fields()

    for field_name in REQUIRED_INDEX_FIELDS:
        assert branches[field_name] == expected[field_name]
        assert isinstance(branches[field_name], int)


def test_includes_cluster_output_branches():
    branches = build_test_branches()

    assert branches["TrackClusterIndex_Z1"] == 2
    assert branches["TrackClusterIndex_Z4"] == 3
    assert branches["TrackClusterIndex_Y1"] == 1
    assert branches["TrackClusterIndex_Y4"] == 2

    assert branches["ClusterMultiplicity_Z1"] == 3
    assert branches["ClusterMultiplicity_Y4"] == 3

    assert branches["NonTrackClusterEnergy_Z1"] == [
        10.0,
        20.0,
    ]


def test_uses_correct_multiplicity_spelling():
    branches = build_test_branches()

    assert "ClusterMultiplicity_Z1" in branches
    assert "ClusterMultiplicity_Y4" in branches

    assert "ClusterMultriplicity_Z1" not in branches
    assert "ClusterMultriplicity_Y4" not in branches


@pytest.mark.parametrize(
    "field_name",
    REQUIRED_TRACK_FIELDS,
)
def test_rejects_missing_track_field(field_name):
    track_fields = valid_track_fields()
    del track_fields[field_name]

    with pytest.raises(
        ValueError,
        match="Missing required track fields",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=track_fields,
            selected_index_fields=valid_index_fields(),
            cluster_output=build_test_cluster_output(),
        )


def test_rejects_unexpected_track_field():
    track_fields = valid_track_fields()
    track_fields["UnexpectedTrackValue"] = 99.0

    with pytest.raises(
        ValueError,
        match="Unexpected track fields",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=track_fields,
            selected_index_fields=valid_index_fields(),
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    "field_name",
    REQUIRED_INDEX_FIELDS,
)
def test_rejects_missing_selected_index_field(field_name):
    index_fields = valid_index_fields()
    del index_fields[field_name]

    with pytest.raises(
        ValueError,
        match="Missing required selected-index fields",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=index_fields,
          cluster_output=build_test_cluster_output(),
        )


def test_rejects_unexpected_selected_index_field():
    index_fields = valid_index_fields()
    index_fields["UnexpectedIndex"] = 2

    with pytest.raises(
        ValueError,
        match="Unexpected selected-index fields",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=index_fields,
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    "field_name",
    REQUIRED_TRACK_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_nonfinite_track_values(
    field_name,
    invalid_value,
):
    track_fields = valid_track_fields()
    track_fields[field_name] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=track_fields,
            selected_index_fields=valid_index_fields(),
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    ("process_id", "event_id"),
    [
        (-1, 345),
        (12, -1),
    ],
)
def test_rejects_negative_event_identity(
    process_id,
    event_id,
):
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=process_id,
                event_id=event_id,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=valid_index_fields(),
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    ("process_id", "event_id"),
    [
        (True, 345),
        (12, False),
        (1.0, 345),
        (12, 345.0),
        ("12", 345),
        (12, "345"),
    ],
)
def test_rejects_noninteger_event_identity(
    process_id,
    event_id,
):
    with pytest.raises(TypeError, match="integer"):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=process_id,
                event_id=event_id,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=valid_index_fields(),
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    "field_name",
    REQUIRED_INDEX_FIELDS,
)
def test_rejects_negative_selected_indices(field_name):
    index_fields = valid_index_fields()
    index_fields[field_name] = -1

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=index_fields,
            cluster_output=build_test_cluster_output(),
        )


@pytest.mark.parametrize(
    "field_name",
    REQUIRED_INDEX_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        1.0,
        "1",
        None,
    ],
)
def test_rejects_noninteger_selected_indices(
    field_name,
    invalid_value,
):
    index_fields = valid_index_fields()
    index_fields[field_name] = invalid_value

    with pytest.raises(TypeError, match="integer"):
        build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=12,
                event_id=345,
            ),
            track_fields=valid_track_fields(),
            selected_index_fields=index_fields,
            cluster_output=build_test_cluster_output(),
        )


def test_output_vector_branches_are_independent():
    branches = build_test_branches()

    z1_energies = branches["NonTrackClusterEnergy_Z1"]
    z2_energies = branches["NonTrackClusterEnergy_Z2"]

    z1_energies.append(999.0)

    assert 999.0 not in z2_energies
