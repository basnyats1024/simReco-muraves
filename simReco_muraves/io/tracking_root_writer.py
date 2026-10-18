# io/tracking_root_writer.py

from __future__ import annotations

from dataclasses import dataclass

from simReco_muraves.models.tracking_event import TrackingEvent

import awkward as ak
import uproot

from simReco_muraves.models.tracking_results import TrackingResult



def _plane_payload(clusters):
    return {
        "energies": [
            c.cluster_energy
            for c in clusters
        ],
        "positions": [
            c.cluster_position
            for c in clusters
        ],
        "sizes": [
            c.cluster_size
            for c in clusters
        ],
        "times": [
            c.cluster_time
            for c in clusters
        ],
        "strip_energies": [
            c.strip_energies
            for c in clusters
        ],
    }


def tracked_records_to_root_payload(
    records: list[TrackedEventRecord],
) -> dict:

    payload = {}

    # ------------------------------------------------------------------
    # Event identity
    # ------------------------------------------------------------------

    payload["ProcessID"] = [r.process_id for r in records]
    payload["Event"] = [r.event for r in records]

    # ------------------------------------------------------------------
    # Multiplicity
    # ------------------------------------------------------------------

    payload["Ntracks_3p_xz"] = [r.Ntracks_3p_xz for r in records]
    payload["Ntracks_3p_xy"] = [r.Ntracks_3p_xy for r in records]

    # ------------------------------------------------------------------
    # 3-point tracks
    # ------------------------------------------------------------------

    payload["Intercept_3p_xz"] = [r.Intercept_3p_xz for r in records]
    payload["Slope_3p_xz"] = [r.Slope_3p_xz for r in records]
    payload["chiSquare_3p_xz"] = [r.chiSquare_3p_xz for r in records]

    payload["Intercept_3p_xy"] = [r.Intercept_3p_xy for r in records]
    payload["Slope_3p_xy"] = [r.Slope_3p_xy for r in records]
    payload["chiSquare_3p_xy"] = [r.chiSquare_3p_xy for r in records]

    payload["track_3p_of_4p_index_xz"] = [
        r.track_3p_of_4p_index_xz
        for r in records
    ]

    payload["track_3p_of_4p_index_xy"] = [
        r.track_3p_of_4p_index_xy
        for r in records
    ]

    payload["track_4p_index_xz"] = [
        r.track_4p_index_xz
        for r in records
    ]

    payload["track_4p_index_xy"] = [
        r.track_4p_index_xy
        for r in records
    ]

    # Plane-4 candidate selection quality.
    #   -1 no promotion, 1 fell back to first candidate, 0 valid Texp.
    payload["best_track_4p_is_texp_null_xz"] = [
        r.best_track_4p_is_texp_null_xz
        for r in records
    ]

    payload["best_track_4p_is_texp_null_xy"] = [
        r.best_track_4p_is_texp_null_xy
        for r in records
    ]


    # ------------------------------------------------------------------
    # Cluster indices
    # ------------------------------------------------------------------

    payload["TrackCluster_z1_index"] = [
        r.TrackCluster_z1_index for r in records
    ]

    payload["TrackCluster_z2_index"] = [
        r.TrackCluster_z2_index for r in records
    ]

    payload["TrackCluster_z3_index"] = [
        r.TrackCluster_z3_index for r in records
    ]

    payload["TrackCluster_y1_index"] = [
        r.TrackCluster_y1_index for r in records
    ]

    payload["TrackCluster_y2_index"] = [
        r.TrackCluster_y2_index for r in records
    ]

    payload["TrackCluster_y3_index"] = [
        r.TrackCluster_y3_index for r in records
    ]

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    payload["Residue_Track3p_z1"] = [
        r.Residue_Track3p_z1 for r in records
    ]

    payload["Residue_Track3p_z2"] = [
        r.Residue_Track3p_z2 for r in records
    ]

    payload["Residue_Track3p_z3"] = [
        r.Residue_Track3p_z3 for r in records
    ]

    payload["Residue_Track3p_y1"] = [
        r.Residue_Track3p_y1 for r in records
    ]

    payload["Residue_Track3p_y2"] = [
        r.Residue_Track3p_y2 for r in records
    ]

    payload["Residue_Track3p_y3"] = [
        r.Residue_Track3p_y3 for r in records
    ]

    # ------------------------------------------------------------------
    # Track energy
    # ------------------------------------------------------------------

    payload["TrackEnergy_3p_xz"] = [
        r.TrackEnergy_3p_xz for r in records
    ]

    payload["TrackEnergy_3p_xy"] = [
        r.TrackEnergy_3p_xy for r in records
    ]

    # ------------------------------------------------------------------
    # Plane 4
    # ------------------------------------------------------------------

    payload["Plane4th_isIntercepted_xz"] = [
        r.Plane4th_isIntercepted_xz for r in records
    ]

    payload["Plane4th_isIntercepted_xy"] = [
        r.Plane4th_isIntercepted_xy for r in records
    ]

    payload["ExpectedPosition_OnPlane4th_xz"] = [
        r.ExpectedPosition_OnPlane4th_xz for r in records
    ]

    payload["ExpectedPosition_OnPlane4th_xy"] = [
        r.ExpectedPosition_OnPlane4th_xy for r in records
    ]

    # ------------------------------------------------------------------
    # Best-track bookkeeping
    # ------------------------------------------------------------------

    payload["BestChi_xz_index"] = [
        r.BestChi_xz_index for r in records
    ]

    payload["BestEnergy_xz_index"] = [
        r.BestEnergy_xz_index for r in records
    ]

    payload["BestChi_xy_index"] = [
        r.BestChi_xy_index for r in records
    ]

    payload["BestEnergy_xy_index"] = [
        r.BestEnergy_xy_index for r in records
    ]

    payload["BestChi_xz"] = [
        r.BestChi_xz for r in records
    ]

    payload["BestEnergy_xz"] = [
        r.BestEnergy_xz for r in records
    ]

    payload["BestChi_xy"] = [
        r.BestChi_xy for r in records
    ]

    payload["BestEnergy_xy"] = [
        r.BestEnergy_xy for r in records
    ]

    # ------------------------------------------------------------------
    # 4-point tracks
    # ------------------------------------------------------------------

    payload["Intercept_4p_xz"] = [
        r.Intercept_4p_xz for r in records
    ]

    payload["Slope_4p_xz"] = [
        r.Slope_4p_xz for r in records
    ]

    payload["chiSquare_4p_xz"] = [
        r.chiSquare_4p_xz for r in records
    ]

    payload["Intercept_4p_xy"] = [
        r.Intercept_4p_xy for r in records
    ]

    payload["Slope_4p_xy"] = [
        r.Slope_4p_xy for r in records
    ]

    payload["chiSquare_4p_xy"] = [
        r.chiSquare_4p_xy for r in records
    ]

    payload["displacement_p4_xz"] = [
        r.displacement_p4_xz for r in records
    ]

    payload["displacement_p4_xy"] = [
        r.displacement_p4_xy for r in records
    ]

    payload["cluster_c4_index_xz"] = [
        r.cluster_c4_index_xz for r in records
    ]

    payload["cluster_c4_index_xy"] = [
        r.cluster_c4_index_xy for r in records
    ]

    payload["ScatteringAngle_xz"] = [
        r.ScatteringAngle_xz for r in records
    ]

    payload["ScatteringAngle_xy"] = [
        r.ScatteringAngle_xy for r in records
    ]

    # ------------------------------------------------------------------
    # Cluster participation
    # ------------------------------------------------------------------

    payload["isInTrack_3p_clZ1"] = [
        r.isInTrack_3p_clZ1 for r in records
    ]

    payload["isInTrack_3p_clZ2"] = [
        r.isInTrack_3p_clZ2 for r in records
    ]

    payload["isInTrack_3p_clZ3"] = [
        r.isInTrack_3p_clZ3 for r in records
    ]

    payload["isInTrack_3p_clY1"] = [
        r.isInTrack_3p_clY1 for r in records
    ]

    payload["isInTrack_3p_clY2"] = [
        r.isInTrack_3p_clY2 for r in records
    ]

    payload["isInTrack_3p_clY3"] = [
        r.isInTrack_3p_clY3 for r in records
    ]

    # ------------------------------------------------------------------
    # Angles
    # ------------------------------------------------------------------

    payload["Theta_3p"] = [
        r.Theta_3p for r in records
    ]

    payload["Phi_3p"] = [
        r.Phi_3p for r in records
    ]

    payload["Theta_4p"] = [
        r.Theta_4p for r in records
    ]

    payload["Phi_4p"] = [
        r.Phi_4p for r in records
    ]

    # ------------------------------------------------------------------
    # Track constituent information
    # ------------------------------------------------------------------

    payload["TrackPosition_z1"] = [
        r.TrackPosition_z1 for r in records
    ]

    payload["TrackPosition_z2"] = [
        r.TrackPosition_z2 for r in records
    ]

    payload["TrackPosition_z3"] = [
        r.TrackPosition_z3 for r in records
    ]

#    payload["TrackPosition_z4"] = [
#        r.TrackPosition_z4 for r in records
#    ]

    payload["TrackPosition_y1"] = [
        r.TrackPosition_y1 for r in records
    ]

    payload["TrackPosition_y2"] = [
        r.TrackPosition_y2 for r in records
    ]

    payload["TrackPosition_y3"] = [
        r.TrackPosition_y3 for r in records
    ]

#    payload["TrackPosition_y4"] = [
#        r.TrackPosition_y4 for r in records
#    ]

    payload["TrackEnergy_z1"] = [
        r.TrackEnergy_z1 for r in records
    ]

    payload["TrackEnergy_z2"] = [
        r.TrackEnergy_z2 for r in records
    ]

    payload["TrackEnergy_z3"] = [
        r.TrackEnergy_z3 for r in records
    ]

#    payload["TrackEnergy_z4"] = [
#        r.TrackEnergy_z4 for r in records
#    ]

    payload["TrackEnergy_y1"] = [
        r.TrackEnergy_y1 for r in records
    ]

    payload["TrackEnergy_y2"] = [
        r.TrackEnergy_y2 for r in records
    ]

    payload["TrackEnergy_y3"] = [
        r.TrackEnergy_y3 for r in records
    ]


#    payload["TrackEnergy_y4"] = [
#        r.TrackEnergy_y4 for r in records
#    ]

    payload["TrackTime_z1"] = [
        r.TrackTime_z1 for r in records
    ]

    payload["TrackTime_z2"] = [
        r.TrackTime_z2 for r in records
    ]

    payload["TrackTime_z3"] = [
        r.TrackTime_z3 for r in records
    ]

#   payload["TrackTime_z4"] = [
#       r.TrackTime_z4 for r in records
#   ]
    
    payload["TrackTime_y1"] = [
        r.TrackTime_y1 for r in records
    ]

    payload["TrackTime_y2"] = [
        r.TrackTime_y2 for r in records
    ]

    payload["TrackTime_y3"] = [
        r.TrackTime_y3 for r in records
    ]

#    payload["TrackTime_y4"] = [
#        r.TrackTime_y4 for r in records
#    ]

    payload["ClusterEnergy_z1"] = [
        r.ClusterEnergy_z1 for r in records
    ]

    payload["ClusterPosition_z1"] = [
        r.ClusterPosition_z1 for r in records
    ]

    payload["ClusterSize_z1"] = [
        r.ClusterSize_z1 for r in records
    ]

    payload["ClusterTime_z1"] = [
        r.ClusterTime_z1 for r in records
    ]

    payload["ClusterStripEnergies_z1"] = [
        r.ClusterStripEnergies_z1 for r in records
    ]

    payload["ClusterEnergy_z2"] = [
        r.ClusterEnergy_z2 for r in records
    ]
    payload["ClusterEnergy_z3"] = [
        r.ClusterEnergy_z3 for r in records
    ]
    payload["ClusterEnergy_z4"] = [
        r.ClusterEnergy_z4 for r in records
    ]

    payload["ClusterEnergy_y1"] = [
        r.ClusterEnergy_y1 for r in records
    ]
    payload["ClusterEnergy_y2"] = [
        r.ClusterEnergy_y2 for r in records
    ]
    payload["ClusterEnergy_y3"] = [
        r.ClusterEnergy_y3 for r in records
    ]
    payload["ClusterEnergy_y4"] = [
        r.ClusterEnergy_y4 for r in records
    ]

    payload["ClusterPosition_z2"] = [
        r.ClusterPosition_z2 for r in records
    ]
    payload["ClusterPosition_z3"] = [
        r.ClusterPosition_z3 for r in records
    ]
    payload["ClusterPosition_z4"] = [
        r.ClusterPosition_z4 for r in records
    ]

    payload["ClusterPosition_y1"] = [
        r.ClusterPosition_y1 for r in records
    ]
    payload["ClusterPosition_y2"] = [
        r.ClusterPosition_y2 for r in records
    ]
    payload["ClusterPosition_y3"] = [
        r.ClusterPosition_y3 for r in records
    ]
    payload["ClusterPosition_y4"] = [
        r.ClusterPosition_y4 for r in records
    ]

    payload["ClusterSize_z2"] = [
        r.ClusterSize_z2 for r in records
    ]
    payload["ClusterSize_z3"] = [
        r.ClusterSize_z3 for r in records
    ]
    payload["ClusterSize_z4"] = [
        r.ClusterSize_z4 for r in records
    ]

    payload["ClusterSize_y1"] = [
        r.ClusterSize_y1 for r in records
    ]
    payload["ClusterSize_y2"] = [
        r.ClusterSize_y2 for r in records
    ]
    payload["ClusterSize_y3"] = [
        r.ClusterSize_y3 for r in records
    ]
    payload["ClusterSize_y4"] = [
        r.ClusterSize_y4 for r in records
    ]

    payload["ClusterTime_z2"] = [
        r.ClusterTime_z2 for r in records
    ]
    payload["ClusterTime_z3"] = [
        r.ClusterTime_z3 for r in records
    ]
    payload["ClusterTime_z4"] = [
        r.ClusterTime_z4 for r in records
    ]

    payload["ClusterTime_y1"] = [
        r.ClusterTime_y1 for r in records
    ]

    payload["ClusterTime_y2"] = [
        r.ClusterTime_y2 for r in records
    ]

    payload["ClusterTime_y3"] = [
        r.ClusterTime_y3 for r in records
    ]

    payload["ClusterTime_y4"] = [
        r.ClusterTime_y4 for r in records
    ]

    payload["ClusterStripEnergies_z2"] = [
        r.ClusterStripEnergies_z2 for r in records
    ]

    payload["ClusterStripEnergies_z3"] = [
        r.ClusterStripEnergies_z3 for r in records
    ]

    payload["ClusterStripEnergies_z4"] = [
        r.ClusterStripEnergies_z4 for r in records
    ]

    payload["ClusterStripEnergies_y1"] = [
        r.ClusterStripEnergies_y1 for r in records
    ]
    payload["ClusterStripEnergies_y2"] = [
        r.ClusterStripEnergies_y2 for r in records
    ]
    payload["ClusterStripEnergies_y3"] = [
        r.ClusterStripEnergies_y3 for r in records
    ]
    payload["ClusterStripEnergies_y4"] = [
        r.ClusterStripEnergies_y4 for r in records
    ]


    return payload


def write_tracked_root(
    output_file: str,
    records: list[TrackedEventRecord],
) -> None:
    """
    Write simulated tracking output to a ROOT file
    with a TTree called 'TrackedData'.
    """

    payload = tracked_records_to_root_payload(records)

    for key, value in payload.items():
        payload[key] = ak.Array(value)

    # --------------------------------------------------------------
    # Find the branch that uproot cannot serialize
    # --------------------------------------------------------------
    """
    for key, value in payload.items():

        print(f"Testing branch: {key}")

        try:

            with uproot.recreate("_debug.root") as f:
                f["test"] = {
                    key: value,
                }

        except Exception as exc:

            print()
            print("=" * 80)
            print("FAILED BRANCH")
            print("=" * 80)

            print("Branch:", key)
            print("Type  :", type(value))
            print("Form  :", value.type)

            raise RuntimeError(
                f"Problematic branch: {key}"
            ) from exc
    """
    # --------------------------------------------------------------
    # Actual output
    # --------------------------------------------------------------

    with uproot.recreate(output_file) as root_file:
        root_file["TrackedData"] = payload

        

def build_tracked_event_record(
    event: TrackingEvent,
    result: TrackingResult,
) -> TrackedEventRecord:

    z1 = _plane_payload(event.z1)
    z2 = _plane_payload(event.z2)
    z3 = _plane_payload(event.z3)
    z4 = _plane_payload(event.z4)

    y1 = _plane_payload(event.y1)
    y2 = _plane_payload(event.y2)
    y3 = _plane_payload(event.y3)
    y4 = _plane_payload(event.y4)


    return TrackedEventRecord(
        process_id=result.process_id,
        event=result.event,

        Ntracks_3p_xz=result.ntracks_3p_xz,
        Ntracks_3p_xy=result.ntracks_3p_xy,

        Intercept_3p_xz=result.tracks_xz.intercepts_3p,
        Slope_3p_xz=result.tracks_xz.slopes_3p,
        chiSquare_3p_xz=result.tracks_xz.chiSquares_3p,

        Intercept_3p_xy=result.tracks_xy.intercepts_3p,
        Slope_3p_xy=result.tracks_xy.slopes_3p,
        chiSquare_3p_xy=result.tracks_xy.chiSquares_3p,

        TrackCluster_z1_index=result.tracks_xz.cluster_index_1,
        TrackCluster_z2_index=result.tracks_xz.cluster_index_2,
        TrackCluster_z3_index=result.tracks_xz.cluster_index_3,

        TrackCluster_y1_index=result.tracks_xy.cluster_index_1,
        TrackCluster_y2_index=result.tracks_xy.cluster_index_2,
        TrackCluster_y3_index=result.tracks_xy.cluster_index_3,

        Residue_Track3p_z1=result.tracks_xz.residue_c1,
        Residue_Track3p_z2=result.tracks_xz.residue_c2,
        Residue_Track3p_z3=result.tracks_xz.residue_c3,

        Residue_Track3p_y1=result.tracks_xy.residue_c1,
        Residue_Track3p_y2=result.tracks_xy.residue_c2,
        Residue_Track3p_y3=result.tracks_xy.residue_c3,

        TrackEnergy_3p_xz=result.tracks_xz.TrackEnergy_3p,
        TrackEnergy_3p_xy=result.tracks_xy.TrackEnergy_3p,


        TrackPosition_z1=result.tracks_xz.track_position_1,
        TrackPosition_z2=result.tracks_xz.track_position_2,
        TrackPosition_z3=result.tracks_xz.track_position_3,

        TrackPosition_y1=result.tracks_xy.track_position_1,
        TrackPosition_y2=result.tracks_xy.track_position_2,
        TrackPosition_y3=result.tracks_xy.track_position_3,

        TrackEnergy_z1=result.tracks_xz.track_energy_1,
        TrackEnergy_z2=result.tracks_xz.track_energy_2,
        TrackEnergy_z3=result.tracks_xz.track_energy_3,

        TrackEnergy_y1=result.tracks_xy.track_energy_1,
        TrackEnergy_y2=result.tracks_xy.track_energy_2,
        TrackEnergy_y3=result.tracks_xy.track_energy_3,

        TrackTime_z1=result.tracks_xz.track_time_1,
        TrackTime_z2=result.tracks_xz.track_time_2,
        TrackTime_z3=result.tracks_xz.track_time_3,

        TrackTime_y1=result.tracks_xy.track_time_1,
        TrackTime_y2=result.tracks_xy.track_time_2,
        TrackTime_y3=result.tracks_xy.track_time_3,

#        TrackPosition_z4=[],
#        TrackPosition_y4=[],

#        TrackEnergy_z4=[],
#        TrackEnergy_y4=[],

#        TrackTime_z4=[],
#        TrackTime_y4=[],

        Plane4th_isIntercepted_xz=result.tracks_xz.Plane4th_isIntercepted,
        Plane4th_isIntercepted_xy=result.tracks_xy.Plane4th_isIntercepted,

        BestChi_xz_index=result.tracks_xz.BestChi_index,
        BestEnergy_xz_index=result.tracks_xz.BestEnergy_index,

        BestChi_xy_index=result.tracks_xy.BestChi_index,
        BestEnergy_xy_index=result.tracks_xy.BestEnergy_index,

        BestChi_xz=result.tracks_xz.BestChi,
        BestEnergy_xz=result.tracks_xz.BestEnergy,

        BestChi_xy=result.tracks_xy.BestChi,
        BestEnergy_xy=result.tracks_xy.BestEnergy,

        track_3p_of_4p_index_xz=
            result.track_3p_of_4p_index_xz,

        track_3p_of_4p_index_xy=
            result.track_3p_of_4p_index_xy,

        track_4p_index_xz=
            result.track_4p_index_xz,

        track_4p_index_xy=
            result.track_4p_index_xy,

        best_track_4p_is_texp_null_xz=
            result.best_track_4p_is_texp_null_xz,

        best_track_4p_is_texp_null_xy=
            result.best_track_4p_is_texp_null_xy,


        ExpectedPosition_OnPlane4th_xz=result.tracks_xz.ExpectedPosition_OnPlane4th,
        ExpectedPosition_OnPlane4th_xy=result.tracks_xy.ExpectedPosition_OnPlane4th,

        Intercept_4p_xz=result.tracks_xz.intercept_4p,
        Slope_4p_xz=result.tracks_xz.slope_4p,
        chiSquare_4p_xz=result.tracks_xz.chiSquares_4p,

        Intercept_4p_xy=result.tracks_xy.intercept_4p,
        Slope_4p_xy=result.tracks_xy.slope_4p,
        chiSquare_4p_xy=result.tracks_xy.chiSquares_4p,

        displacement_p4_xz=result.tracks_xz.displacement_p4,
        displacement_p4_xy=result.tracks_xy.displacement_p4,

        cluster_c4_index_xz=result.tracks_xz.cluster_indices_4,
        cluster_c4_index_xy=result.tracks_xy.cluster_indices_4,

        ScatteringAngle_xz=result.tracks_xz.ScatteringAngles,
        ScatteringAngle_xy=result.tracks_xy.ScatteringAngles,

        isInTrack_3p_clZ1=result.tracks_xz.IsInTrack_clusters1,
        isInTrack_3p_clZ2=result.tracks_xz.IsInTrack_clusters2,
        isInTrack_3p_clZ3=result.tracks_xz.IsInTrack_clusters3,

        isInTrack_3p_clY1=result.tracks_xy.IsInTrack_clusters1,
        isInTrack_3p_clY2=result.tracks_xy.IsInTrack_clusters2,
        isInTrack_3p_clY3=result.tracks_xy.IsInTrack_clusters3,

        Theta_3p=result.theta_3p,
        Phi_3p=result.phi_3p,

        Theta_4p=result.theta_4p,
        Phi_4p=result.phi_4p,

        ClusterEnergy_z1=z1["energies"],
        ClusterEnergy_z2=z2["energies"],
        ClusterEnergy_z3=z3["energies"],
        ClusterEnergy_z4=z4["energies"],

        ClusterEnergy_y1=y1["energies"],
        ClusterEnergy_y2=y2["energies"],
        ClusterEnergy_y3=y3["energies"],
        ClusterEnergy_y4=y4["energies"],

        ClusterPosition_z1=z1["positions"],
        ClusterPosition_z2=z2["positions"],
        ClusterPosition_z3=z3["positions"],
        ClusterPosition_z4=z4["positions"],

        ClusterPosition_y1=y1["positions"],
        ClusterPosition_y2=y2["positions"],
        ClusterPosition_y3=y3["positions"],
        ClusterPosition_y4=y4["positions"],

        ClusterSize_z1=z1["sizes"],
        ClusterSize_z2=z2["sizes"],
        ClusterSize_z3=z3["sizes"],
        ClusterSize_z4=z4["sizes"],

        ClusterSize_y1=y1["sizes"],
        ClusterSize_y2=y2["sizes"],
        ClusterSize_y3=y3["sizes"],
        ClusterSize_y4=y4["sizes"],

        ClusterTime_z1=z1["times"],
        ClusterTime_z2=z2["times"],
        ClusterTime_z3=z3["times"],
        ClusterTime_z4=z4["times"],

        ClusterTime_y1=y1["times"],
        ClusterTime_y2=y2["times"],
        ClusterTime_y3=y3["times"],
        ClusterTime_y4=y4["times"],

        ClusterStripEnergies_z1=z1["strip_energies"],
        ClusterStripEnergies_z2=z2["strip_energies"],
        ClusterStripEnergies_z3=z3["strip_energies"],
        ClusterStripEnergies_z4=z4["strip_energies"],

        ClusterStripEnergies_y1=y1["strip_energies"],
        ClusterStripEnergies_y2=y2["strip_energies"],
        ClusterStripEnergies_y3=y3["strip_energies"],
        ClusterStripEnergies_y4=y4["strip_energies"],


    )

@dataclass
class TrackedEventRecord:

    process_id: int
    event: int

    Ntracks_3p_xz: int
    Ntracks_3p_xy: int

    Intercept_3p_xz: list[float]
    Slope_3p_xz: list[float]
    chiSquare_3p_xz: list[float]

    Intercept_3p_xy: list[float]
    Slope_3p_xy: list[float]
    chiSquare_3p_xy: list[float]

    TrackCluster_z1_index: list[int]
    TrackCluster_z2_index: list[int]
    TrackCluster_z3_index: list[int]

    TrackCluster_y1_index: list[int]
    TrackCluster_y2_index: list[int]
    TrackCluster_y3_index: list[int]

    Residue_Track3p_z1: list[float]
    Residue_Track3p_z2: list[float]
    Residue_Track3p_z3: list[float]

    Residue_Track3p_y1: list[float]
    Residue_Track3p_y2: list[float]
    Residue_Track3p_y3: list[float]

    TrackEnergy_3p_xz: list[float]
    TrackEnergy_3p_xy: list[float]

    Plane4th_isIntercepted_xz: list[int]
    Plane4th_isIntercepted_xy: list[int]

    BestChi_xz_index: int
    BestEnergy_xz_index: int

    BestChi_xy_index: int
    BestEnergy_xy_index: int

    BestChi_xz: float
    BestEnergy_xz: float

    BestChi_xy: float
    BestEnergy_xy: float

    track_3p_of_4p_index_xz: int
    track_3p_of_4p_index_xy: int

    track_4p_index_xz: int
    track_4p_index_xy: int

    # Plane-4 candidate selection quality; see TrackingResult.
    best_track_4p_is_texp_null_xz: int
    best_track_4p_is_texp_null_xy: int

    ExpectedPosition_OnPlane4th_xz: list[float]
    ExpectedPosition_OnPlane4th_xy: list[float]

    Intercept_4p_xz: list[list[float]]
    Slope_4p_xz: list[list[float]]
    chiSquare_4p_xz: list[list[float]]

    Intercept_4p_xy: list[list[float]]
    Slope_4p_xy: list[list[float]]
    chiSquare_4p_xy: list[list[float]]

    displacement_p4_xz: list[list[float]]
    displacement_p4_xy: list[list[float]]

    cluster_c4_index_xz: list[list[float]]
    cluster_c4_index_xy: list[list[float]]

    ScatteringAngle_xz: list[list[float]]
    ScatteringAngle_xy: list[list[float]]

    isInTrack_3p_clZ1: list[int]
    isInTrack_3p_clZ2: list[int]
    isInTrack_3p_clZ3: list[int]

    isInTrack_3p_clY1: list[int]
    isInTrack_3p_clY2: list[int]
    isInTrack_3p_clY3: list[int]

    Theta_3p: float
    Phi_3p: float

    Theta_4p: float
    Phi_4p: float

    # ------------------------------------------------------------------
    # Track constituent positions
    # ------------------------------------------------------------------

    TrackPosition_z1: list[float]
    TrackPosition_z2: list[float]
    TrackPosition_z3: list[float]

    TrackPosition_y1: list[float]
    TrackPosition_y2: list[float]
    TrackPosition_y3: list[float]

    # ------------------------------------------------------------------
    # Track constituent energies
    # ------------------------------------------------------------------

    TrackEnergy_z1: list[float]
    TrackEnergy_z2: list[float]
    TrackEnergy_z3: list[float]

    TrackEnergy_y1: list[float]
    TrackEnergy_y2: list[float]
    TrackEnergy_y3: list[float]

    # ------------------------------------------------------------------
    # Track constituent times
    # ------------------------------------------------------------------

    TrackTime_z1: list[float]
    TrackTime_z2: list[float]
    TrackTime_z3: list[float]

    TrackTime_y1: list[float]
    TrackTime_y2: list[float]
    TrackTime_y3: list[float]

    # ------------------------------------------------------------------
    # Plane 4 constituent information
    # ------------------------------------------------------------------

#    TrackPosition_z4: list[float]
#    TrackPosition_y4: list[float]

#    TrackEnergy_z4: list[float]
#    TrackEnergy_y4: list[float]

#    TrackTime_z4: list[float]
#    TrackTime_y4: list[float]


    # -----------------------------------------------------------------------------
    # Relevant Cluster Info so that trackedData tree can also do gTrk selection
    # (i.e., without requiring clusteredData tree)
    # ----------------------------------------------------------------------------
    ClusterEnergy_z1: list[float]
    ClusterEnergy_z2: list[float]
    ClusterEnergy_z3: list[float]
    ClusterEnergy_z4: list[float]

    ClusterEnergy_y1: list[float]
    ClusterEnergy_y2: list[float]
    ClusterEnergy_y3: list[float]
    ClusterEnergy_y4: list[float]

    ClusterPosition_z1: list[float]
    ClusterPosition_z2: list[float]
    ClusterPosition_z3: list[float]
    ClusterPosition_z4: list[float]

    ClusterPosition_y1: list[float]
    ClusterPosition_y2: list[float]
    ClusterPosition_y3: list[float]
    ClusterPosition_y4: list[float]

    ClusterSize_z1: list[int]
    ClusterSize_z2: list[int]
    ClusterSize_z3: list[int]
    ClusterSize_z4: list[int]

    ClusterSize_y1: list[int]
    ClusterSize_y2: list[int]
    ClusterSize_y3: list[int]
    ClusterSize_y4: list[int]

    ClusterTime_z1: list[float]
    ClusterTime_z2: list[float]
    ClusterTime_z3: list[float]
    ClusterTime_z4: list[float]

    ClusterTime_y1: list[float]
    ClusterTime_y2: list[float]
    ClusterTime_y3: list[float]
    ClusterTime_y4: list[float]

    ClusterStripEnergies_z1: list[list[float]]
    ClusterStripEnergies_z2: list[list[float]]
    ClusterStripEnergies_z3: list[list[float]]
    ClusterStripEnergies_z4: list[list[float]]

    ClusterStripEnergies_y1: list[list[float]]
    ClusterStripEnergies_y2: list[list[float]]
    ClusterStripEnergies_y3: list[list[float]]
    ClusterStripEnergies_y4: list[list[float]]
