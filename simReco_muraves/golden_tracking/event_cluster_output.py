from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Mapping, Sequence



class GoldenConsistencyCode(IntEnum):
    OK = 0

    INVALID_3P_XZ_INDEX = 101
    INVALID_3P_XY_INDEX = 102

    INVALID_4P_PARENT_XZ = 201
    INVALID_4P_PARENT_XY = 202

    INVALID_4P_INNER_XZ = 301
    INVALID_4P_INNER_XY = 302

    INVALID_P4_CLUSTER_INDEX_XZ = 401
    INVALID_P4_CLUSTER_INDEX_XY = 402


@dataclass(frozen=True)
class GoldenIndexValidation:
    valid: bool
    code: GoldenConsistencyCode
    message: str = ""


def _valid_index(index: int, values: Sequence[object]) -> bool:
    return isinstance(index, int) and 0 <= index < len(values)


def _valid_nested_index(
    parent_index: int,
    inner_index: int,
    values: Sequence[Sequence[object]],
) -> bool:
    return (
        _valid_index(parent_index, values)
        and isinstance(inner_index, int)
        and 0 <= inner_index < len(values[parent_index])
    )


def validated_cluster_index(
    value: int | float,
    cluster_count: int,
) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        index = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        index = int(value)
    else:
        return None

    if not 0 <= index < cluster_count:
        return None

    return index


def validate_selected_indices(
    *,
    best_3p_xz_index: int,
    best_3p_xy_index: int,
    cluster_index_1_xz: Sequence[int],
    cluster_index_1_xy: Sequence[int],
    parent_4p_xz: int,
    parent_4p_xy: int,
    inner_4p_xz: int,
    inner_4p_xy: int,
    cluster_indices_4_xz: Sequence[Sequence[int | float]],
    cluster_indices_4_xy: Sequence[Sequence[int | float]],
    n_clusters_p4_xz: int,
    n_clusters_p4_xy: int,
) -> GoldenIndexValidation:
    if not _valid_index(best_3p_xz_index, cluster_index_1_xz):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_3P_XZ_INDEX,
            "Invalid selected 3p XZ index.",
        )

    if not _valid_index(best_3p_xy_index, cluster_index_1_xy):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_3P_XY_INDEX,
            "XZ golden gate passed, but selected 3p XY state is invalid.",
        )

    if not _valid_index(parent_4p_xz, cluster_indices_4_xz):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_4P_PARENT_XZ,
            "Invalid XZ compact 4p-parent row.",
        )

    if not _valid_index(parent_4p_xy, cluster_indices_4_xy):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_4P_PARENT_XY,
            "Invalid XY compact 4p-parent row.",
        )

    if not _valid_nested_index(
        parent_4p_xz,
        inner_4p_xz,
        cluster_indices_4_xz,
    ):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_4P_INNER_XZ,
            "Invalid XZ inner 4p candidate index.",
        )

    if not _valid_nested_index(
        parent_4p_xy,
        inner_4p_xy,
        cluster_indices_4_xy,
    ):
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_4P_INNER_XY,
            "Invalid XY inner 4p candidate index.",
        )

    p4_cluster_xz = validated_cluster_index(
        cluster_indices_4_xz[parent_4p_xz][inner_4p_xz],
        n_clusters_p4_xz,
    )
    if p4_cluster_xz is None:
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_P4_CLUSTER_INDEX_XZ,
            "Invalid selected plane-4 XZ cluster index.",
        )

    p4_cluster_xy = validated_cluster_index(
        cluster_indices_4_xy[parent_4p_xy][inner_4p_xy],
        n_clusters_p4_xy,
    )
    if p4_cluster_xy is None:
        return GoldenIndexValidation(
            False,
            GoldenConsistencyCode.INVALID_P4_CLUSTER_INDEX_XY,
            "Invalid selected plane-4 XY cluster index.",
        )

    return GoldenIndexValidation(
        True,
        GoldenConsistencyCode.OK,
    )


@dataclass(frozen=True)
class SelectedClusterIndices:
    z1: int
    z2: int
    z3: int
    z4: int

    y1: int
    y2: int
    y3: int
    y4: int


@dataclass(frozen=True)
class SelectedClusterResolution:
    valid: bool
    indices: SelectedClusterIndices | None = None
    code: GoldenConsistencyCode = GoldenConsistencyCode.OK
    message: str = ""


def _resolve_3p_cluster_index(
    *,
    selected_track_index: int,
    candidate_cluster_indices: Sequence[int],
    cluster_count: int,
) -> int | None:
    if not isinstance(selected_track_index, int):
        return None

    if not 0 <= selected_track_index < len(candidate_cluster_indices):
        return None

    return validated_cluster_index(
        candidate_cluster_indices[selected_track_index],
        cluster_count,
    )


def _resolve_4p_cluster_index(
    *,
    parent_index: int,
    inner_index: int,
    candidate_cluster_indices: Sequence[Sequence[int | float]],
    cluster_count: int,
) -> int | None:
    if not isinstance(parent_index, int):
        return None

    if not isinstance(inner_index, int):
        return None

    if not 0 <= parent_index < len(candidate_cluster_indices):
        return None

    row = candidate_cluster_indices[parent_index]

    if not 0 <= inner_index < len(row):
        return None

    return validated_cluster_index(
        row[inner_index],
        cluster_count,
    )


def resolve_selected_cluster_indices(
    *,
    best_3p_xz_index: int,
    best_3p_xy_index: int,
    parent_4p_xz: int,
    parent_4p_xy: int,
    inner_4p_xz: int,
    inner_4p_xy: int,
    cluster_index_1_xz: Sequence[int],
    cluster_index_2_xz: Sequence[int],
    cluster_index_3_xz: Sequence[int],
    cluster_index_1_xy: Sequence[int],
    cluster_index_2_xy: Sequence[int],
    cluster_index_3_xy: Sequence[int],
    cluster_indices_4_xz: Sequence[Sequence[int | float]],
    cluster_indices_4_xy: Sequence[Sequence[int | float]],
    n_clusters_z1: int,
    n_clusters_z2: int,
    n_clusters_z3: int,
    n_clusters_z4: int,
    n_clusters_y1: int,
    n_clusters_y2: int,
    n_clusters_y3: int,
    n_clusters_y4: int,
) -> SelectedClusterResolution:
    z1 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xz_index,
        candidate_cluster_indices=cluster_index_1_xz,
        cluster_count=n_clusters_z1,
    )
    if z1 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XZ_INDEX,
            message="Invalid selected Z1 cluster index.",
        )

    z2 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xz_index,
        candidate_cluster_indices=cluster_index_2_xz,
        cluster_count=n_clusters_z2,
    )
    if z2 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XZ_INDEX,
            message="Invalid selected Z2 cluster index.",
        )

    z3 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xz_index,
        candidate_cluster_indices=cluster_index_3_xz,
        cluster_count=n_clusters_z3,
    )
    if z3 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XZ_INDEX,
            message="Invalid selected Z3 cluster index.",
        )

    y1 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xy_index,
        candidate_cluster_indices=cluster_index_1_xy,
        cluster_count=n_clusters_y1,
    )
    if y1 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XY_INDEX,
            message="Invalid selected Y1 cluster index.",
        )

    y2 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xy_index,
        candidate_cluster_indices=cluster_index_2_xy,
        cluster_count=n_clusters_y2,
    )
    if y2 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XY_INDEX,
            message="Invalid selected Y2 cluster index.",
        )

    y3 = _resolve_3p_cluster_index(
        selected_track_index=best_3p_xy_index,
        candidate_cluster_indices=cluster_index_3_xy,
        cluster_count=n_clusters_y3,
    )
    if y3 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_3P_XY_INDEX,
            message="Invalid selected Y3 cluster index.",
        )

    z4 = _resolve_4p_cluster_index(
        parent_index=parent_4p_xz,
        inner_index=inner_4p_xz,
        candidate_cluster_indices=cluster_indices_4_xz,
        cluster_count=n_clusters_z4,
    )
    if z4 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_P4_CLUSTER_INDEX_XZ,
            message="Invalid selected Z4 cluster index.",
        )

    y4 = _resolve_4p_cluster_index(
        parent_index=parent_4p_xy,
        inner_index=inner_4p_xy,
        candidate_cluster_indices=cluster_indices_4_xy,
        cluster_count=n_clusters_y4,
    )
    if y4 is None:
        return SelectedClusterResolution(
            valid=False,
            code=GoldenConsistencyCode.INVALID_P4_CLUSTER_INDEX_XY,
            message="Invalid selected Y4 cluster index.",
        )

    return SelectedClusterResolution(
        valid=True,
        indices=SelectedClusterIndices(
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            y1=y1,
            y2=y2,
            y3=y3,
            y4=y4,
        ),
        code=GoldenConsistencyCode.OK,
    )


@dataclass(frozen=True)
class SelectedClusterOutput:
    index: int
    energy: float
    position: float
    size: int
    texp: float
    maximum_strip_energy: float
    minimum_strip_energy: float


@dataclass(frozen=True)
class PlaneClusterOutput:
    selected: SelectedClusterOutput
    nontrack: NonTrackClusters
    multiplicity: int


def build_plane_cluster_output(
    *,
    energies: Sequence[float],
    positions: Sequence[float],
    sizes: Sequence[int],
    texp: Sequence[float],
    strip_energies: Sequence[Sequence[float]],
    selected_index: int,
) -> PlaneClusterOutput:
    collection_lengths = {
        len(energies),
        len(positions),
        len(sizes),
        len(texp),
        len(strip_energies),
    }

    if len(collection_lengths) != 1:
        raise ValueError(
            "All plane cluster collections must have the same length"
        )

    cluster_count = len(energies)

    if isinstance(selected_index, bool):
        raise TypeError("selected_index must be an integer")

    if not isinstance(selected_index, int):
        raise TypeError("selected_index must be an integer")

    if not 0 <= selected_index < cluster_count:
        raise IndexError(
            f"selected_index {selected_index} is outside "
            f"the valid range [0, {cluster_count})"
        )

    selected_size = sizes[selected_index]

    if isinstance(selected_size, bool):
        raise TypeError("selected cluster size must be an integer")

    if not isinstance(selected_size, int):
        raise TypeError("selected cluster size must be an integer")

    strip_summary = summarize_strip_energies(
        strip_energies=strip_energies[selected_index],
        cluster_size=selected_size,
    )

    selected = SelectedClusterOutput(
        index=selected_index,
        energy=float(energies[selected_index]),
        position=float(positions[selected_index]),
        size=selected_size,
        texp=float(texp[selected_index]),
        maximum_strip_energy=strip_summary.maximum,
        minimum_strip_energy=strip_summary.minimum,
    )

    nontrack = extract_nontrack_clusters(
        energies=energies,
        positions=positions,
        sizes=sizes,
        selected_index=selected_index,
    )

    return PlaneClusterOutput(
        selected=selected,
        nontrack=nontrack,
        multiplicity=cluster_count,
    )


PLANE_NAMES = (
    "z1",
    "z2",
    "z3",
    "z4",
    "y1",
    "y2",
    "y3",
    "y4",
)


@dataclass(frozen=True)
class PlaneClusterInput:
    energies: Sequence[float]
    positions: Sequence[float]
    sizes: Sequence[int]
    texp: Sequence[float]
    strip_energies: Sequence[Sequence[float]]


@dataclass(frozen=True)
class GoldenEventClusterOutput:
    z1: PlaneClusterOutput
    z2: PlaneClusterOutput
    z3: PlaneClusterOutput
    z4: PlaneClusterOutput

    y1: PlaneClusterOutput
    y2: PlaneClusterOutput
    y3: PlaneClusterOutput
    y4: PlaneClusterOutput


def build_golden_event_cluster_output(
    *,
    selected: SelectedClusterIndices,
    planes: Mapping[str, PlaneClusterInput],
) -> GoldenEventClusterOutput:
    missing_planes = [
        plane_name
        for plane_name in PLANE_NAMES
        if plane_name not in planes
    ]

    if missing_planes:
        raise ValueError(
            "Missing required plane inputs: "
            + ", ".join(missing_planes)
        )

    unexpected_planes = [
        plane_name
        for plane_name in planes
        if plane_name not in PLANE_NAMES
    ]

    if unexpected_planes:
        raise ValueError(
            "Unexpected plane inputs: "
            + ", ".join(unexpected_planes)
        )

    outputs: dict[str, PlaneClusterOutput] = {}

    for plane_name in PLANE_NAMES:
        plane_input = planes[plane_name]
        selected_index = getattr(selected, plane_name)

        outputs[plane_name] = build_plane_cluster_output(
            energies=plane_input.energies,
            positions=plane_input.positions,
            sizes=plane_input.sizes,
            texp=plane_input.texp,
            strip_energies=plane_input.strip_energies,
            selected_index=selected_index,
        )

    return GoldenEventClusterOutput(
        z1=outputs["z1"],
        z2=outputs["z2"],
        z3=outputs["z3"],
        z4=outputs["z4"],
        y1=outputs["y1"],
        y2=outputs["y2"],
        y3=outputs["y3"],
        y4=outputs["y4"],
    )
@dataclass(frozen=True)
class NonTrackClusters:
    energies: list[float]
    positions: list[float]
    sizes: list[int]


def extract_nontrack_clusters(
    *,
    energies: Sequence[float],
    positions: Sequence[float],
    sizes: Sequence[int],
    selected_index: int,
) -> NonTrackClusters:
    if not (
        len(energies)
        == len(positions)
        == len(sizes)
    ):
        raise ValueError(
            "energies, positions, and sizes must have the same length"
        )

    output_energies: list[float] = []
    output_positions: list[float] = []
    output_sizes: list[int] = []

    selected_index_is_valid = (
        isinstance(selected_index, int)
        and not isinstance(selected_index, bool)
        and 0 <= selected_index < len(energies)
    )

    for index, (energy, position, size) in enumerate(
        zip(energies, positions, sizes)
    ):
        if selected_index_is_valid and index == selected_index:
            continue

        output_energies.append(float(energy))
        output_positions.append(float(position))
        output_sizes.append(int(size))

    return NonTrackClusters(
        energies=output_energies,
        positions=output_positions,
        sizes=output_sizes,
    )
@dataclass(frozen=True)
class StripEnergySummary:
    maximum: float
    minimum: float
    count: int
    is_descending: bool


def summarize_strip_energies(
    *,
    strip_energies: Sequence[float],
    cluster_size: int,
) -> StripEnergySummary:
    if isinstance(cluster_size, bool) or not isinstance(cluster_size, int):
        raise TypeError("cluster_size must be an integer")

    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")

    if len(strip_energies) != cluster_size:
        raise ValueError(
            "strip-energy list length must match cluster_size"
        )

    energies = [float(value) for value in strip_energies]

    if not all(math.isfinite(value) for value in energies):
        raise ValueError("strip energies must be finite")

    is_descending = all(
        energies[index] >= energies[index + 1]
        for index in range(len(energies) - 1)
    )

    if not is_descending:
        raise ValueError(
            "strip energies must be ordered from highest to lowest"
        )

    return StripEnergySummary(
        maximum=energies[0],
        minimum=energies[-1],
        count=len(energies),
        is_descending=True,
    )
