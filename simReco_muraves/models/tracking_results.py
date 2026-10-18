# models/tracking_result.py

from __future__ import annotations

from dataclasses import dataclass

from simReco_muraves.tracking.Tracking import TracksCollection


@dataclass
class TrackingResult:

    process_id: int
    event: int

    tracks_xz: TracksCollection
    tracks_xy: TracksCollection

    ntracks_3p_xz: int
    ntracks_3p_xy: int

    best_track_xz_index: int
    best_track_xy_index: int

    theta_3p: float = -1.0
    phi_3p: float = -1.0

    theta_4p: float = -1.0
    phi_4p: float = -1.0

    track_3p_of_4p_index_xz: int = -1
    track_3p_of_4p_index_xy: int = -1

    track_4p_index_xz: int = -1
    track_4p_index_xy: int = -1

    best_track_3p_chi_square_xz: float = -1.0
    best_track_3p_chi_square_xy: float = -1.0

    best_track_4p_chi_square_xz: float = -1.0
    best_track_4p_chi_square_xy: float = -1.0

    # Plane-4 candidate selection quality.
    #
    #   -1 : no four-plane promotion happened for this event
    #    1 : promoted, but no plane-4 candidate had a valid Texp, so the
    #        selection fell back to the first candidate
    #    0 : promoted with a valid Texp
    #
    # Mirrors best_track_4p_is_texp_null_* in the experimental
    # MURAVES_reco.py, which records the same distinction.
    best_track_4p_is_texp_null_xz: int = -1
    best_track_4p_is_texp_null_xy: int = -1
