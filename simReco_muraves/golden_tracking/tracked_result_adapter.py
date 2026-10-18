from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TracksCollectionAdapter:
    """Subset of TracksCollection needed by the golden projection.

    Field names mirror TracksCollection so that the golden record
    builder can consume either this adapter (tracked input) or the real
    collection (clustered input) without branching.
    """

    cluster_index_1: list[int]
    cluster_index_2: list[int]
    cluster_index_3: list[int]

    cluster_indices_4: list[list[float]]

    ScatteringAngles: list[list[float]]

    #: Nested four-plane fit quality, laid out like cluster_indices_4.
    chiSquares_4p: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class TrackingResultAdapter:
    process_id: int
    event: int

    tracks_xz: TracksCollectionAdapter
    tracks_xy: TracksCollectionAdapter

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

    # Plane-4 candidate selection quality; see TrackingResult.
    # Defaults to -1 so that TrackedData files written before these
    # branches existed still load.
    best_track_4p_is_texp_null_xz: int = -1
    best_track_4p_is_texp_null_xy: int = -1
