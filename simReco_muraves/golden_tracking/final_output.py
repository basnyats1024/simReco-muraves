from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .event_cluster_output import (
    GoldenEventClusterOutput,
    PLANE_NAMES,
)
from .golden_selector import GoldenSelection


REQUIRED_TRACK_FIELDS = (
    "Theta",
    "Phi",
    "Scattering_xz",
    "Scattering_xy",
    "ChiSquare_xz",
    "ChiSquare_xy",
    "ChiSquare_3p_xz",
    "ChiSquare_3p_xy",
)

REQUIRED_INDEX_FIELDS = (
    "BestTrack3pIndex_xz",
    "BestTrack3pIndex_xy",
    "Track3pOf4pRow_xz",
    "Track3pOf4pRow_xy",
    "Track4pInnerIndex_xz",
    "Track4pInnerIndex_xy",
)

# Selection-quality flags. Unlike the index fields these may legally
# be -1, so they are validated separately rather than with minimum=0.
#
#   -1 no four-plane promotion happened
#    1 promoted, but no plane-4 candidate had a valid Texp
#    0 promoted with a valid Texp
REQUIRED_FLAG_FIELDS = (
    "BestTrack4pIsTexpNull_xz",
    "BestTrack4pIsTexpNull_xy",
)

DEFAULT_FLAG_FIELDS = {
    "BestTrack4pIsTexpNull_xz": -1,
    "BestTrack4pIsTexpNull_xy": -1,
}


@dataclass(frozen=True)
class GoldenEventIdentity:
    process_id: int
    event_id: int


def _validated_integer(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer"
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{field_name} must be greater than or equal to "
            f"{minimum}"
        )

    return value


def _validated_finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be a real number"
        )

    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{field_name} must be a real number"
        ) from error

    if not math.isfinite(converted):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return converted


def _validate_exact_fields(
    *,
    values: Mapping[str, object],
    required_fields: tuple[str, ...],
    collection_name: str,
) -> None:
    missing = [
        field
        for field in required_fields
        if field not in values
    ]

    if missing:
        raise ValueError(
            f"Missing required {collection_name} fields: "
            + ", ".join(missing)
        )

    unexpected = [
        field
        for field in values
        if field not in required_fields
    ]

    if unexpected:
        raise ValueError(
            f"Unexpected {collection_name} fields: "
            + ", ".join(unexpected)
        )


def build_final_golden_branches(
    *,
    identity: GoldenEventIdentity,
    track_fields: Mapping[str, float],
    selected_index_fields: Mapping[str, int],
    cluster_output: GoldenEventClusterOutput,
    flag_fields: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    process_id = _validated_integer(
        identity.process_id,
        field_name="ProcessID",
        minimum=0,
    )

    event_id = _validated_integer(
        identity.event_id,
        field_name="EventID",
        minimum=0,
    )

    _validate_exact_fields(
        values=track_fields,
        required_fields=REQUIRED_TRACK_FIELDS,
        collection_name="track",
    )

    _validate_exact_fields(
        values=selected_index_fields,
        required_fields=REQUIRED_INDEX_FIELDS,
        collection_name="selected-index",
    )

    if flag_fields is None:
        flag_fields = DEFAULT_FLAG_FIELDS

    _validate_exact_fields(
        values=flag_fields,
        required_fields=REQUIRED_FLAG_FIELDS,
        collection_name="flag",
    )

    branches: dict[str, Any] = {
        "ProcessID": process_id,
        "EventID": event_id,
    }

    for field_name in REQUIRED_TRACK_FIELDS:
        branches[field_name] = _validated_finite_float(
            track_fields[field_name],
            field_name=field_name,
        )

    for field_name in REQUIRED_INDEX_FIELDS:
        branches[field_name] = _validated_integer(
            selected_index_fields[field_name],
            field_name=field_name,
            minimum=0,
        )

    for field_name in REQUIRED_FLAG_FIELDS:
        branches[field_name] = _validated_integer(
            flag_fields[field_name],
            field_name=field_name,
            minimum=-1,
        )

    cluster_branches = build_cluster_output_branches(
        cluster_output
    )

    duplicate_fields = set(branches).intersection(
        cluster_branches
    )

    if duplicate_fields:
        raise ValueError(
            "Duplicate output branch names: "
            + ", ".join(sorted(duplicate_fields))
        )

    branches.update(cluster_branches)

    return branches
@dataclass(frozen=True)
class GoldenTrackRecord:
    theta: float
    phi: float

    scattering_xz: float
    scattering_xy: float

    chi_square_4p_xz: float
    chi_square_4p_xy: float

    chi_square_3p_xz: float
    chi_square_3p_xy: float


def build_golden_track_record(
    event: Any,
    selection: GoldenSelection,
) -> GoldenTrackRecord | None:
    """Extract selected track quantities using reference indices."""

    if not selection.accepted:
        return None

    p3_xz = selection.best_3p_index_xz
    p3_xy = selection.best_3p_index_xy

    parent_xz = selection.parent_3p_index_xz
    parent_xy = selection.parent_3p_index_xy

    extension_xz = selection.extension_4p_index_xz
    extension_xy = selection.extension_4p_index_xy

    return GoldenTrackRecord(
        theta=event.theta_4p,
        phi=event.phi_4p,
        scattering_xz=event.scattering_angle_xz[parent_xz][extension_xz],
        scattering_xy=event.scattering_angle_xy[parent_xy][extension_xy],
        chi_square_4p_xz=event.chi_square_4p_xz[parent_xz][extension_xz],
        chi_square_4p_xy=event.chi_square_4p_xy[parent_xy][extension_xy],
        chi_square_3p_xz=event.chi_square_3p_xz[p3_xz],
        chi_square_3p_xy=event.chi_square_3p_xy[p3_xy],
    )
def build_cluster_output_branches(
    output: GoldenEventClusterOutput,
) -> dict[str, Any]:
    branches: dict[str, Any] = {}

    for plane_name in PLANE_NAMES:
        plane_output = getattr(output, plane_name)

        view = plane_name[0].upper()
        plane_number = plane_name[1]
        suffix = f"{view}{plane_number}"

        branches[f"TrackClusterIndex_{suffix}"] = (
            plane_output.selected.index
        )
        branches[f"TrackClusterEnergy_{suffix}"] = (
            plane_output.selected.energy
        )
        branches[f"TrackClusterPosition_{suffix}"] = (
            plane_output.selected.position
        )
        branches[f"TrackClusterSize_{suffix}"] = (
            plane_output.selected.size
        )
        branches[f"TrackClusterTexp_{suffix}"] = (
            plane_output.selected.texp
        )
        branches[f"TrackClusterMaxStripEnergy_{suffix}"] = (
            plane_output.selected.maximum_strip_energy
        )
        branches[f"TrackClusterMinStripEnergy_{suffix}"] = (
            plane_output.selected.minimum_strip_energy
        )

        branches[f"ClusterMultiplicity_{suffix}"] = (
            plane_output.multiplicity
        )

        branches[f"NonTrackClusterEnergy_{suffix}"] = list(
            plane_output.nontrack.energies
        )
        branches[f"NonTrackClusterPosition_{suffix}"] = list(
            plane_output.nontrack.positions
        )
        branches[f"NonTrackClusterSize_{suffix}"] = list(
            plane_output.nontrack.sizes
        )

    return branches
