from __future__ import annotations

import pytest

from simReco_muraves.golden_tracking.event_cluster_output import (
    GoldenEventClusterOutput,
    PlaneClusterInput,
    build_golden_event_cluster_output,
)
from simReco_muraves.golden_tracking.event_cluster_output import SelectedClusterIndices


def make_plane(
    *,
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


def valid_planes() -> dict[str, PlaneClusterInput]:
    return {
        "z1": make_plane(base=10.0, count=3),
        "z2": make_plane(base=20.0, count=4),
        "z3": make_plane(base=30.0, count=5),
        "z4": make_plane(base=40.0, count=4),
        "y1": make_plane(base=50.0, count=2),
        "y2": make_plane(base=60.0, count=3),
        "y3": make_plane(base=70.0, count=4),
        "y4": make_plane(base=80.0, count=3),
    }


def valid_selected_indices() -> SelectedClusterIndices:
    return SelectedClusterIndices(
        z1=2,
        z2=3,
        z3=4,
        z4=3,
        y1=1,
        y2=2,
        y3=3,
        y4=2,
    )


def test_builds_outputs_for_all_eight_planes():
    result = build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=valid_planes(),
    )

    assert isinstance(result, GoldenEventClusterOutput)

    assert result.z1.selected.index == 2
    assert result.z2.selected.index == 3
    assert result.z3.selected.index == 4
    assert result.z4.selected.index == 3

    assert result.y1.selected.index == 1
    assert result.y2.selected.index == 2
    assert result.y3.selected.index == 3
    assert result.y4.selected.index == 2


def test_selected_values_come_from_correct_planes():
    planes = valid_planes()

    result = build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=planes,
    )

    assert result.z1.selected.energy == planes["z1"].energies[2]
    assert result.z2.selected.energy == planes["z2"].energies[3]
    assert result.z3.selected.energy == planes["z3"].energies[4]
    assert result.z4.selected.energy == planes["z4"].energies[3]

    assert result.y1.selected.energy == planes["y1"].energies[1]
    assert result.y2.selected.energy == planes["y2"].energies[2]
    assert result.y3.selected.energy == planes["y3"].energies[3]
    assert result.y4.selected.energy == planes["y4"].energies[2]


def test_records_correct_multiplicities():
    result = build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=valid_planes(),
    )

    assert result.z1.multiplicity == 3
    assert result.z2.multiplicity == 4
    assert result.z3.multiplicity == 5
    assert result.z4.multiplicity == 4

    assert result.y1.multiplicity == 2
    assert result.y2.multiplicity == 3
    assert result.y3.multiplicity == 4
    assert result.y4.multiplicity == 3


def test_each_nontrack_collection_excludes_selected_cluster():
    planes = valid_planes()

    result = build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=planes,
    )

    for plane_name in (
        "z1",
        "z2",
        "z3",
        "z4",
        "y1",
        "y2",
        "y3",
        "y4",
    ):
        plane_result = getattr(result, plane_name)
        plane_input = planes[plane_name]

        expected_energies = [
            float(energy)
            for index, energy in enumerate(plane_input.energies)
            if index != plane_result.selected.index
        ]

        expected_positions = [
            float(position)
            for index, position in enumerate(plane_input.positions)
            if index != plane_result.selected.index
        ]

        expected_sizes = [
            int(size)
            for index, size in enumerate(plane_input.sizes)
            if index != plane_result.selected.index
        ]

        assert plane_result.nontrack.energies == expected_energies
        assert plane_result.nontrack.positions == expected_positions
        assert plane_result.nontrack.sizes == expected_sizes


def test_selected_strip_energy_summaries_are_correct():
    planes = valid_planes()

    result = build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=planes,
    )

    for plane_name in (
        "z1",
        "z2",
        "z3",
        "z4",
        "y1",
        "y2",
        "y3",
        "y4",
    ):
        plane_result = getattr(result, plane_name)
        plane_input = planes[plane_name]
        selected_index = plane_result.selected.index

        selected_strip_energies = (
            plane_input.strip_energies[selected_index]
        )

        assert (
            plane_result.selected.maximum_strip_energy
            == selected_strip_energies[0]
        )
        assert (
            plane_result.selected.minimum_strip_energy
            == selected_strip_energies[-1]
        )


def test_rejects_missing_plane_input():
    planes = valid_planes()
    del planes["y4"]

    with pytest.raises(
        ValueError,
        match="Missing required plane inputs: y4",
    ):
        build_golden_event_cluster_output(
            selected=valid_selected_indices(),
            planes=planes,
        )


def test_reportsall_missing_planes():
    planes = valid_planes()
    del planes["z4"]
    del planes["y3"]

    with pytest.raises(ValueError) as error:
        build_golden_event_cluster_output(
            selected=valid_selected_indices(),
            planes=planes,
        )

    message = str(error.value)

    assert "z4" in message
    assert "y3" in message


def test_rejects_unexpected_plane_input():
    planes = valid_planes()
    planes["x1"] = make_plane(
        base=90.0,
        count=2,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected plane inputs: x1",
    ):
        build_golden_event_cluster_output(
            selected=valid_selected_indices(),
            planes=planes,
        )


@pytest.mark.parametrize(
    ("selected_field", "invalid_index"),
    [
        ("z1", -1),
        ("z2", 4),
        ("z3", 5),
        ("z4", 4),
        ("y1", 2),
        ("y2", 3),
        ("y3", 4),
        ("y4", 3),
    ],
)
def test_rejects_invalid_selected_index_for_each_plane(
    selected_field,
    invalid_index,
):
    selected_values = {
        "z1": 2,
        "z2": 3,
        "z3": 4,
        "z4": 3,
        "y1": 1,
        "y2": 2,
        "y3": 3,
        "y4": 2,
    }

    selected_values[selected_field] = invalid_index
    selected = SelectedClusterIndices(**selected_values)

    with pytest.raises(IndexError):
        build_golden_event_cluster_output(
            selected=selected,
            planes=valid_planes(),
        )


def test_rejects_invalid_data_in_any_plane():
    planes = valid_planes()

    planes["z3"] = PlaneClusterInput(
        energies=planes["z3"].energies,
        positions=planes["z3"].positions,
        sizes=planes["z3"].sizes,
        texp=planes["z3"].texp,
        strip_energies=[
            [30.0],
            [40.0, 39.0],
            [50.0, 49.0, 48.0],
            [60.0, 59.0, 58.0, 57.0],
            [70.0, 80.0, 60.0, 50.0, 40.0],
        ],
    )

    with pytest.raises(
        ValueError,
        match="highest to lowest",
    ):
        build_golden_event_cluster_output(
            selected=valid_selected_indices(),
            planes=planes,
        )


def test_input_plane_mapping_is_not_modified():
    planes = valid_planes()
    original_keys = tuple(planes.keys())

    build_golden_event_cluster_output(
        selected=valid_selected_indices(),
        planes=planes,
    )

    assert tuple(planes.keys()) == original_keys
