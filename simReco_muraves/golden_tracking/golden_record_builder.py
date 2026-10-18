# simReco_muraves/golden_tracking/golden_record_builder.py

"""Single construction path for a golden-track record.

Two routes reach the golden projection:

    clustered input -> tracking in memory -> golden
    tracked input   -> persisted tracking state -> golden

Both consume the same selected tracking state and must produce the same
record, the same event gate and the same diagnostics. Keeping one
builder is what makes that true by construction rather than by
convention; the two routes previously held independent copies of this
logic and had drifted in both the gate condition and the four-plane
chi-square.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from ..nested_access import MISSING_VALUE, nested_value_strict
from .event_cluster_output import (
    PLANE_NAMES,
    PlaneClusterInput,
    build_golden_event_cluster_output,
    resolve_selected_cluster_indices,
)
from .final_output import (
    GoldenEventIdentity,
    build_final_golden_branches,
)
from .golden_pipeline import (
    GoldenProcessingResult,
    consistency_error_result,
    rejected_result,
    selected_result,
)


class SelectedTracks(Protocol):
    """Per-projection candidate state needed by the golden projection."""

    cluster_index_1: Sequence[int]
    cluster_index_2: Sequence[int]
    cluster_index_3: Sequence[int]
    cluster_indices_4: Sequence[Sequence[Any]]
    ScatteringAngles: Sequence[Sequence[float]]


class SelectedTrackingState(Protocol):
    """Everything the golden record needs from a tracking result.

    Satisfied by both TrackingResult (clustered route) and
    TrackingResultAdapter (tracked route).
    """

    process_id: int
    event: int

    tracks_xz: SelectedTracks
    tracks_xy: SelectedTracks

    best_track_xz_index: int
    best_track_xy_index: int

    track_3p_of_4p_index_xz: int
    track_3p_of_4p_index_xy: int

    track_4p_index_xz: int
    track_4p_index_xy: int

    theta_4p: float
    phi_4p: float

    best_track_3p_chi_square_xz: float
    best_track_3p_chi_square_xy: float

    best_track_4p_chi_square_xz: float
    best_track_4p_chi_square_xy: float


def build_golden_record(
    *,
    planes: Mapping[str, PlaneClusterInput],
    result: SelectedTrackingState,
) -> GoldenProcessingResult:
    """Project one event's selected tracking state into a golden record.

    The event gate is the experimental XZ gate: a valid selected
    three-plane XZ fit and a valid promoted four-plane XZ fit. Invalid
    XY state is a reconstruction-consistency error rather than a normal
    physics rejection, which preserves the experimental semantics while
    still refusing to dereference a bad index.
    """
    missing_planes = [
        plane_name
        for plane_name in PLANE_NAMES
        if plane_name not in planes
    ]

    if missing_planes:
        return consistency_error_result(
            "missing_plane_inputs:" + ",".join(missing_planes)
        )

    if result.best_track_3p_chi_square_xz == MISSING_VALUE:
        return rejected_result("invalid_3p_xz")

    if result.best_track_4p_chi_square_xz == MISSING_VALUE:
        return rejected_result("invalid_4p_xz")

    resolution = resolve_selected_cluster_indices(
        best_3p_xz_index=result.best_track_xz_index,
        best_3p_xy_index=result.best_track_xy_index,
        parent_4p_xz=result.track_3p_of_4p_index_xz,
        parent_4p_xy=result.track_3p_of_4p_index_xy,
        inner_4p_xz=result.track_4p_index_xz,
        inner_4p_xy=result.track_4p_index_xy,
        cluster_index_1_xz=result.tracks_xz.cluster_index_1,
        cluster_index_2_xz=result.tracks_xz.cluster_index_2,
        cluster_index_3_xz=result.tracks_xz.cluster_index_3,
        cluster_index_1_xy=result.tracks_xy.cluster_index_1,
        cluster_index_2_xy=result.tracks_xy.cluster_index_2,
        cluster_index_3_xy=result.tracks_xy.cluster_index_3,
        cluster_indices_4_xz=result.tracks_xz.cluster_indices_4,
        cluster_indices_4_xy=result.tracks_xy.cluster_indices_4,
        n_clusters_z1=len(planes["z1"].energies),
        n_clusters_z2=len(planes["z2"].energies),
        n_clusters_z3=len(planes["z3"].energies),
        n_clusters_z4=len(planes["z4"].energies),
        n_clusters_y1=len(planes["y1"].energies),
        n_clusters_y2=len(planes["y2"].energies),
        n_clusters_y3=len(planes["y3"].energies),
        n_clusters_y4=len(planes["y4"].energies),
    )

    if not resolution.valid or resolution.indices is None:
        return consistency_error_result(
            f"selected_cluster_resolution_{int(resolution.code)}"
        )

    try:
        scattering_xz = nested_value_strict(
            result.tracks_xz.ScatteringAngles,
            result.track_3p_of_4p_index_xz,
            result.track_4p_index_xz,
            "Scattering_xz",
        )

        scattering_xy = nested_value_strict(
            result.tracks_xy.ScatteringAngles,
            result.track_3p_of_4p_index_xy,
            result.track_4p_index_xy,
            "Scattering_xy",
        )

        cluster_output = build_golden_event_cluster_output(
            selected=resolution.indices,
            planes=planes,
        )

        record = build_final_golden_branches(
            identity=GoldenEventIdentity(
                process_id=result.process_id,
                event_id=result.event,
            ),
            track_fields={
                "Theta": result.theta_4p,
                "Phi": result.phi_4p,
                "Scattering_xz": scattering_xz,
                "Scattering_xy": scattering_xy,
                "ChiSquare_xz": result.best_track_4p_chi_square_xz,
                "ChiSquare_xy": result.best_track_4p_chi_square_xy,
                "ChiSquare_3p_xz": result.best_track_3p_chi_square_xz,
                "ChiSquare_3p_xy": result.best_track_3p_chi_square_xy,
            },
            selected_index_fields={
                "BestTrack3pIndex_xz": result.best_track_xz_index,
                "BestTrack3pIndex_xy": result.best_track_xy_index,
                "Track3pOf4pRow_xz": result.track_3p_of_4p_index_xz,
                "Track3pOf4pRow_xy": result.track_3p_of_4p_index_xy,
                "Track4pInnerIndex_xz": result.track_4p_index_xz,
                "Track4pInnerIndex_xy": result.track_4p_index_xy,
            },
            flag_fields={
                "BestTrack4pIsTexpNull_xz": getattr(
                    result,
                    "best_track_4p_is_texp_null_xz",
                    -1,
                ),
                "BestTrack4pIsTexpNull_xy": getattr(
                    result,
                    "best_track_4p_is_texp_null_xy",
                    -1,
                ),
            },
            cluster_output=cluster_output,
        )
    except (IndexError, TypeError, ValueError) as error:
        return consistency_error_result(
            f"golden_record_build_failure:{type(error).__name__}"
        )

    return selected_result(record)
