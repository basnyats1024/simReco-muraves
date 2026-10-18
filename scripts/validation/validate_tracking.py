#!/usr/bin/env python3

import os
from pathlib import Path

import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIGURATION
# ============================================================

DATA_FILE = (
    "/Users/sbasnet/Documents/MURAVES_stuff/"
    "digiWork/srcDigi/dataFiles/expData/"
    #"MURAVES_AnalyzedData_ROSSO_run3780-3818_v0.root"
    "MURAVES_AnalyzedData_NERO_run6780-6835.root"
    #"MURAVES_AnalyzedData_BLU_run8820-88655.root"
)

MC_FILE = (
    "/Users/sbasnet/Documents/MURAVES_stuff/"
    "digiWork/srcDigi/dataFiles/simData/TMP/"
    "musimData_run4_tracked.root"
)

DATA_TREE = "AnalyzedData"
MC_TREE = "TrackedData"

OUTDIR = "/Users/sbasnet/Documents/MURAVES_stuff/digiWork/srcDigi/simAna/tracking_validation"


# ============================================================
# HELPERS
# ============================================================

def flatten(arr):
    return ak.to_numpy(
        ak.flatten(arr, axis=None)
    )


def get_branch(tree, name):

    arr = tree[name].array(
        library="ak"
    )

    return flatten(arr)


def positive_only(arr):
    return arr[arr > 0]


def finite_only(arr):
    return arr[np.isfinite(arr)]


def make_comparison_plot(
    data,
    mc,
    title,
    xlabel,
    outfile,
    bins=100,
):

    data = finite_only(data)
    mc = finite_only(mc)

    plt.figure(figsize=(8, 5))

    if "ScatteringAngle" in title:
        plt.xlim(-25.0, 25.0)

    plt.hist(
        data,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Data",
    )

    plt.hist(
        mc,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="MC",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized entries")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTDIR,
            outfile,
        ),
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {outfile}"
    )


def print_stats(
    name,
    data,
    mc,
):

    data = finite_only(data)
    mc = finite_only(mc)

    print(
        f"{name:20s}"
        f" Data N={len(data):10,d}"
        f" Mean={np.mean(data):8.3f}"
        f" RMS={np.std(data):8.3f}"
    )

    print(
        f"{'':20s}"
        f" MC   N={len(mc):10,d}"
        f" Mean={np.mean(mc):8.3f}"
        f" RMS={np.std(mc):8.3f}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTDIR,
        exist_ok=True,
    )

    data_tree = uproot.open(
        DATA_FILE
    )[DATA_TREE]

    mc_tree = uproot.open(
        MC_FILE
    )[MC_TREE]

    print()
    print("====================================")
    print("TRACKING VALIDATION")
    print("====================================")
    print()

    # ========================================================
    # Multiplicity
    # ========================================================

    ntracks_xz_data = get_branch(
        data_tree,
        "Ntracks_3p_xz",
    )

    ntracks_xz_mc = get_branch(
        mc_tree,
        "Ntracks_3p_xz",
    )

    ntracks_xy_data = get_branch(
        data_tree,
        "Ntracks_3p_xy",
    )

    ntracks_xy_mc = get_branch(
        mc_tree,
        "Ntracks_3p_xy",
    )

    make_comparison_plot(
        ntracks_xz_data,
        ntracks_xz_mc,
        "Track Multiplicity XZ",
        "Ntracks_3p_xz",
        "multiplicity_xz.png",
        bins=np.arange(-0.5, 8.5, 1.0),
    )

    make_comparison_plot(
        ntracks_xy_data,
        ntracks_xy_mc,
        "Track Multiplicity XY",
        "Ntracks_3p_xy",
        "multiplicity_xy.png",
        bins=np.arange(-0.5, 8.5, 1.0),
    )

    print_stats(
        "Multiplicity XZ",
        ntracks_xz_data,
        ntracks_xz_mc,
    )

    print_stats(
        "Multiplicity XY",
        ntracks_xy_data,
        ntracks_xy_mc,
    )

    # ========================================================
    # Theta
    # ========================================================

    theta3_data = positive_only(
        get_branch(
            data_tree,
            "Theta_3p",
        )
    )

    theta3_mc = positive_only(
        get_branch(
            mc_tree,
            "Theta_3p",
        )
    )

    theta4_data = positive_only(
        get_branch(
            data_tree,
            "Theta_4p",
        )
    )

    theta4_mc = positive_only(
        get_branch(
            mc_tree,
            "Theta_4p",
        )
    )

    make_comparison_plot(
        theta3_data,
        theta3_mc,
        "Theta 3p",
        r"$\theta_{3p}$ [deg]",
        "theta3p.png",
    )

    make_comparison_plot(
        theta4_data,
        theta4_mc,
        "Theta 4p",
        r"$\theta_{4p}$ [deg]",
        "theta4p.png",
    )

    print_stats(
        "Theta_3p",
        theta3_data,
        theta3_mc,
    )

    print_stats(
        "Theta_4p",
        theta4_data,
        theta4_mc,
    )

    # ========================================================
    # Phi
    # ========================================================

    phi3_data = positive_only(
        get_branch(
            data_tree,
            "Phi_3p",
        )
    )

    phi3_mc = positive_only(
        get_branch(
            mc_tree,
            "Phi_3p",
        )
    )

    phi4_data = positive_only(
        get_branch(
            data_tree,
            "Phi_4p",
        )
    )

    phi4_mc = positive_only(
        get_branch(
            mc_tree,
            "Phi_4p",
        )
    )

    make_comparison_plot(
        phi3_data,
        phi3_mc,
        "Phi 3p",
        r"$\phi_{3p}$ [deg]",
        "phi3p.png",
    )

    make_comparison_plot(
        phi4_data,
        phi4_mc,
        "Phi 4p",
        r"$\phi_{4p}$ [deg]",
        "phi4p.png",
    )

    print_stats(
        "Phi_3p",
        phi3_data,
        phi3_mc,
    )

    print_stats(
        "Phi_4p",
        phi4_data,
        phi4_mc,
    )

    # ========================================================
    # Chi2
    # ========================================================

    for branch in [
        "BestChi_xz",
        "BestChi_xy",
    ]:

        data_vals = positive_only(
            get_branch(data_tree, branch)
        )

        mc_vals = positive_only(
            get_branch(mc_tree, branch)
        )

        data_vals = data_vals[data_vals < 100]
        mc_vals = mc_vals[mc_vals < 100]

        make_comparison_plot(
            data_vals,
            mc_vals,
            branch,
            branch,
            f"{branch}.png",
        )

    # ========================================================
    # Residuals
    # ========================================================

    for branch in [
        "Residue_Track3p_z1",
        "Residue_Track3p_z2",
        "Residue_Track3p_z3",
        "Residue_Track3p_y1",
        "Residue_Track3p_y2",
        "Residue_Track3p_y3",
    ]:

        make_comparison_plot(
            get_branch(
                data_tree,
                branch,
            ),
            get_branch(
                mc_tree,
                branch,
            ),
            branch,
            branch,
            f"{branch}.png",
        )

    # ========================================================
    # Scattering Angles
    # ========================================================

    for branch in [
        "ScatteringAngle_xz",
        "ScatteringAngle_xy",
    ]:

        data_vals = finite_only(
            get_branch(
                data_tree,
                branch,
            )
        )

        mc_vals = finite_only(
            get_branch(
                mc_tree,
                branch,
            )
        )

        make_comparison_plot(
            data_vals,
            mc_vals,
            branch,
            branch,
            f"{branch}.png",
            bins=100
        )

        print_stats(
            branch,
            data_vals,
            mc_vals,
        )

    # ========================================================
    # 4P Acceptance
    # ========================================================

    frac_data = (
        len(theta4_data)
        / len(theta3_data)
        * 100.0
    )

    frac_mc = (
        len(theta4_mc)
        / len(theta3_mc)
        * 100.0
    )

    print()
    print("====================================")
    print("4P ACCEPTANCE")
    print("====================================")

    print(
        f"Data : {frac_data:.2f}%"
    )

    print(
        f"MC   : {frac_mc:.2f}%"
    )

    print()
    print("Finished.")
    print()


if __name__ == "__main__":
    main()
