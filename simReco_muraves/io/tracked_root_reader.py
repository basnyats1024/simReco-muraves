# simReco_muraves/io/tracked_root_reader.py

"""Read persisted tracking output back into golden-projection inputs.

TrackedData carries both the per-plane cluster payload and the selected
tracking state, so the golden projection can run from it without
re-running tracking. The adapters built here are deliberately
field-compatible with TrackingEvent/TrackingResult so that both routes
feed the same golden record builder.
"""

from __future__ import annotations

import awkward as ak
import uproot

from simReco_muraves.golden_tracking.event_cluster_output import (
    PLANE_NAMES,
    PlaneClusterInput,
)
from simReco_muraves.golden_tracking.tracked_result_adapter import (
    TracksCollectionAdapter,
    TrackingResultAdapter,
)
from simReco_muraves.nested_access import nested_value_or_missing


TRACKED_TREE_NAME = "TrackedData"

#: Previous name for the plane cluster payload type. The golden builder
#: now takes PlaneClusterInput directly; this alias keeps older callers
#: and tests importing from here working.
PlaneClusterInputData = PlaneClusterInput


def load_tracked_tree(path: str):
    """Return the TrackedData tree.

    The caller owns the open file; prefer load_tracked_events() unless
    direct tree access is needed.
    """
    root_file = uproot.open(path)
    return root_file[TRACKED_TREE_NAME]


def _read_arrays(path: str):
    with uproot.open(path) as root_file:
        tree = root_file[TRACKED_TREE_NAME]
        return tree.arrays(library="ak")


def _has_field(arrays, name: str) -> bool:
    return name in arrays.fields


def _plane_input(
    arrays,
    plane: str,
    index: int,
) -> PlaneClusterInput:

    return PlaneClusterInput(
        energies=ak.to_list(
            arrays[f"ClusterEnergy_{plane}"][index]
        ),
        positions=ak.to_list(
            arrays[f"ClusterPosition_{plane}"][index]
        ),
        sizes=ak.to_list(
            arrays[f"ClusterSize_{plane}"][index]
        ),
        texp=ak.to_list(
            arrays[f"ClusterTime_{plane}"][index]
        ),
        strip_energies=ak.to_list(
            arrays[f"ClusterStripEnergies_{plane}"][index]
        ),
    )


def _planes_at(arrays, index: int) -> dict[str, PlaneClusterInput]:
    return {
        plane_name: _plane_input(arrays, plane_name, index)
        for plane_name in PLANE_NAMES
    }


def _optional_int(arrays, name: str, index: int, default: int) -> int:
    """Read an integer branch that older TrackedData files may lack."""
    if not _has_field(arrays, name):
        return default

    return int(arrays[name][index])


def _tracks_adapter(
    arrays,
    index: int,
    *,
    plane_prefix: str,
    projection: str,
) -> TracksCollectionAdapter:

    chi_branch = f"chiSquare_4p_{projection}"

    return TracksCollectionAdapter(
        cluster_index_1=ak.to_list(
            arrays[f"TrackCluster_{plane_prefix}1_index"][index]
        ),
        cluster_index_2=ak.to_list(
            arrays[f"TrackCluster_{plane_prefix}2_index"][index]
        ),
        cluster_index_3=ak.to_list(
            arrays[f"TrackCluster_{plane_prefix}3_index"][index]
        ),
        cluster_indices_4=ak.to_list(
            arrays[f"cluster_c4_index_{projection}"][index]
        ),
        ScatteringAngles=ak.to_list(
            arrays[f"ScatteringAngle_{projection}"][index]
        ),
        chiSquares_4p=ak.to_list(arrays[chi_branch][index]),
    )


def _result_adapter(arrays, index: int) -> TrackingResultAdapter:

    tracks_xz = _tracks_adapter(
        arrays,
        index,
        plane_prefix="z",
        projection="xz",
    )

    tracks_xy = _tracks_adapter(
        arrays,
        index,
        plane_prefix="y",
        projection="xy",
    )

    parent_4p_xz = int(arrays["track_3p_of_4p_index_xz"][index])
    parent_4p_xy = int(arrays["track_3p_of_4p_index_xy"][index])

    inner_4p_xz = int(arrays["track_4p_index_xz"][index])
    inner_4p_xy = int(arrays["track_4p_index_xy"][index])

    best_3p_index_xz = int(arrays["BestChi_xz_index"][index])
    best_3p_index_xy = int(arrays["BestChi_xy_index"][index])

    best_3p_chi_xz = -1.0
    best_3p_chi_xy = -1.0

    if best_3p_index_xz >= 0:
        best_3p_chi_xz = float(arrays["BestChi_xz"][index])

    if best_3p_index_xy >= 0:
        best_3p_chi_xy = float(arrays["BestChi_xy"][index])

    # Recovered with the same bounds check the clustered route applies
    # in reconstruction_adapter, so both routes gate on the same value.
    best_4p_chi_xz = nested_value_or_missing(
        tracks_xz.chiSquares_4p,
        parent_4p_xz,
        inner_4p_xz,
    )

    best_4p_chi_xy = nested_value_or_missing(
        tracks_xy.chiSquares_4p,
        parent_4p_xy,
        inner_4p_xy,
    )

    return TrackingResultAdapter(
        process_id=int(arrays["ProcessID"][index]),
        event=int(arrays["Event"][index]),
        tracks_xz=tracks_xz,
        tracks_xy=tracks_xy,
        best_track_xz_index=best_3p_index_xz,
        best_track_xy_index=best_3p_index_xy,
        track_3p_of_4p_index_xz=parent_4p_xz,
        track_3p_of_4p_index_xy=parent_4p_xy,
        track_4p_index_xz=inner_4p_xz,
        track_4p_index_xy=inner_4p_xy,
        theta_4p=float(arrays["Theta_4p"][index]),
        phi_4p=float(arrays["Phi_4p"][index]),
        best_track_3p_chi_square_xz=best_3p_chi_xz,
        best_track_3p_chi_square_xy=best_3p_chi_xy,
        best_track_4p_chi_square_xz=best_4p_chi_xz,
        best_track_4p_chi_square_xy=best_4p_chi_xy,
        best_track_4p_is_texp_null_xz=_optional_int(
            arrays,
            "best_track_4p_is_texp_null_xz",
            index,
            -1,
        ),
        best_track_4p_is_texp_null_xy=_optional_int(
            arrays,
            "best_track_4p_is_texp_null_xy",
            index,
            -1,
        ),
    )


def load_tracked_events(
    path: str,
) -> list[tuple[dict[str, PlaneClusterInput], TrackingResultAdapter]]:
    """Load (plane payload, tracking state) pairs in ROOT entry order.

    Reads the file once, unlike calling load_plane_inputs() and
    load_tracking_result_adapters() separately.
    """
    arrays = _read_arrays(path)

    return [
        (_planes_at(arrays, index), _result_adapter(arrays, index))
        for index in range(len(arrays))
    ]


def load_plane_inputs(
    path: str,
) -> list[dict[str, PlaneClusterInput]]:
    """Cluster payload for all eight planes, per entry."""
    arrays = _read_arrays(path)

    return [
        _planes_at(arrays, index)
        for index in range(len(arrays))
    ]


def load_tracking_result_adapters(
    path: str,
) -> list[TrackingResultAdapter]:
    """Selected tracking state, per entry."""
    arrays = _read_arrays(path)

    return [
        _result_adapter(arrays, index)
        for index in range(len(arrays))
    ]
