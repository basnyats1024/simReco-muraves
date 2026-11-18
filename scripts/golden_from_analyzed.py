#!/usr/bin/env python3
"""Golden four-plane selection on AnalyzedData _v1.

A one-off reimplementation of AnalysisOfCluster_and_tracks4planes.C for
the _v1 reconstruction, whose four-plane candidate branches are flat
(double[]) rather than nested (vector<vector<double>>), so the macro
cannot read them as it stands.

    python golden_from_analyzed.py INPUT.root [-o OUTPUT.root] [--max-events N]

Selection, as in the macro:

    keep events with BestTrack_3p_ChiSquare_xz != -1 and
                     BestTrack_4p_ChiSquare_xz != -1
    planes 1-3: the selected cluster is TrackCluster_<plane>_index[BestTrack_3p_<view>_index]
    plane 4:    the selected cluster is cluster_c4_index of the selected 4p candidate
    every other cluster on a plane is a non-track cluster, in its original order

Finding the selected 4p candidate in the flat arrays: it is the one entry
whose chi-square and scattering angle equal the stored scalars
BestTrack_4p_ChiSquare_<view> and BestTracks_ScatteringAngle_<view>. This
needs no assumption about how the flat arrays were nested. As a
cross-check, the report also shows how often the position agrees with the
rule "each candidate row has one entry per plane-4 cluster", i.e.
parent * Nclusters_plane4 + inner.

Differences from the macro, all safety checks it does not make:

    events whose indices fall outside their arrays are excluded and counted
    events without a unique 4p match, in either view, are excluded and counted
    events whose strip count differs from their summed cluster sizes are excluded

Output: a Tracks4ptree TTree with the macro's branch names, including the
legacy TrackClusterExp_* and ClusterMultriplicity_* spellings. Run,
TriggerRate and WorkingPoint carry over; timestamp replaces Datetime;
RunDuration is not available in _v1 and is left out.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


PLANES = ["Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4"]
VIEW = {"Z": "xz", "Y": "xy"}

TREE_IN = "AnalyzedData"
TREE_OUT = "Tracks4ptree"

#: Exact equality is expected (the scalar is a copy of the array entry);
#: the tolerance only absorbs a float/double round trip.
MATCH_RTOL = 1e-9
MATCH_ATOL = 1e-12


# ==========================================================
# Reading
# ==========================================================


class Jagged:
    """One list per event, stored flat."""

    def __init__(self, counts: np.ndarray, values: np.ndarray):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.values = np.asarray(values)
        self.starts = np.concatenate(([0], np.cumsum(self.counts)[:-1])).astype(np.int64)
        self.event = np.repeat(np.arange(len(self.counts)), self.counts)

    def pick(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(flat position, valid) of element index[event] in each event's list."""
        index = np.asarray(index)
        valid = np.isfinite(index) & (index >= 0) & (index < self.counts)
        position = np.where(valid, self.starts + np.where(valid, index, 0).astype(np.int64), 0)
        return position, valid


def read_scalar(tree, name, start, stop) -> np.ndarray:
    return np.asarray(tree[name].array(library="np", entry_start=start, entry_stop=stop))


def read_jagged(tree, name, start, stop) -> Jagged:
    array = tree[name].array(library="ak", entry_start=start, entry_stop=stop)
    return Jagged(ak.to_numpy(ak.num(array, axis=1)), ak.to_numpy(ak.flatten(array, axis=None)))


# ==========================================================
# Selection on one chunk of events
# ==========================================================


def select_chunk(tree, start, stop, reasons: Counter, rule: Counter):
    """The output columns for the selected events in [start, stop)."""
    n = stop - start
    S = lambda name: read_scalar(tree, name, start, stop)
    J = lambda name: read_jagged(tree, name, start, stop)

    ok = (S("BestTrack_3p_ChiSquare_xz") != -1) & (S("BestTrack_4p_ChiSquare_xz") != -1)
    reasons["pass XZ gate"] += int(ok.sum())

    def exclude(mask, reason):
        bad = ok & ~mask
        if bad.any():
            reasons[reason] += int(bad.sum())
        return ok & mask

    out = {}
    best_3p = {view: S(f"BestTrack_3p_{view}_index") for view in ("xz", "xy")}

    # ---------- best 3p chi-square ----------
    for view in ("xz", "xy"):
        chi3 = J(f"chiSquare_3p_{view}")
        position, valid = chi3.pick(best_3p[view])
        ok = exclude(valid, f"best 3p index out of range ({view})")
        out[f"ChiSquare_3p_{view}"] = np.where(valid, chi3.values[position] if len(chi3.values) else np.nan, np.nan)

    # ---------- selected 4p candidate, found through the stored scalars ----------
    selected_c4 = {}
    for view, plane in (("xz", "Z4"), ("xy", "Y4")):
        chi4 = J(f"chiSquare_4p_{view}")
        scat = J(f"ScatteringAngle_{view}")
        c4 = J(f"cluster_c4_index_{view}")

        aligned = np.array_equal(chi4.counts, scat.counts) and np.array_equal(chi4.counts, c4.counts)
        if not aligned:
            same = (chi4.counts == scat.counts) & (chi4.counts == c4.counts)
            ok = exclude(same, f"4p arrays of different length ({view})")

        chi_scalar = S(f"BestTrack_4p_ChiSquare_{view}").astype(float)
        scat_scalar = S(f"BestTracks_ScatteringAngle_{view}").astype(float)

        event = chi4.event
        match = np.isclose(chi4.values, chi_scalar[event], rtol=MATCH_RTOL, atol=MATCH_ATOL)
        if len(scat.values) == len(chi4.values):
            scat_target = scat_scalar[event]
            match &= np.isclose(scat.values, scat_target, rtol=MATCH_RTOL, atol=MATCH_ATOL) | ~np.isfinite(scat_target)

        n_match = np.bincount(event[match], minlength=n)
        unique = n_match == 1
        ok = exclude(unique, f"no unique 4p candidate matching the scalars ({view})")

        position = np.zeros(n, dtype=np.int64)
        matched = np.flatnonzero(match)
        keep = unique[event[match]]
        position[event[match][keep]] = matched[keep]

        # cross-check: does the row rule land on the same entry?
        n_plane4 = J(f"ClusterSize_{plane}").counts
        parent = S(f"Track_3p_of_4p_index_{view}").astype(np.int64)
        inner = S(f"Track_4p_index_{view}").astype(np.int64)
        rule_position = chi4.starts + parent * n_plane4 + inner
        checked = ok & unique
        rule[f"{view} checked"] += int(checked.sum())
        rule[f"{view} agrees"] += int((checked & (rule_position == position)).sum())

        out[f"ChiSquare_{view}"] = np.where(unique, chi4.values[position] if len(chi4.values) else np.nan, np.nan)
        out[f"Scattering_{view}"] = np.where(unique, scat.values[position] if len(scat.values) else np.nan, np.nan)
        selected_c4[plane] = np.where(unique, np.rint(c4.values[position]) if len(c4.values) else -1, -1).astype(np.int64)

    # ---------- per plane: selected cluster, strips, non-track clusters ----------
    plane_columns = {}
    for plane in PLANES:
        view = VIEW[plane[0]]
        size = J(f"ClusterSize_{plane}")
        energy = J(f"ClusterEnergy_{plane}")
        position_ = J(f"ClusterPosition_{plane}")
        texp = J(f"Cluster{plane}_Texp")
        strips = J(f"StripsEnergy_{plane}")

        if plane.endswith("4"):
            selected = selected_c4[plane]
        else:
            track_cluster = J(f"TrackCluster_{plane.lower()}_index")
            where, valid = track_cluster.pick(best_3p[view])
            ok = exclude(valid, f"best 3p index outside TrackCluster_{plane.lower()}_index")
            selected = np.where(valid, track_cluster.values[where] if len(track_cluster.values) else -1, -1).astype(np.int64)

        cluster, valid = size.pick(selected)
        ok = exclude(valid, f"selected cluster index out of range ({plane})")

        summed = np.bincount(size.event, weights=size.values, minlength=n) if len(size.values) else np.zeros(n)
        ok = exclude(strips.counts == np.rint(summed).astype(np.int64), f"strip count differs from cluster sizes ({plane})")

        # first strip of each cluster, counted within its event
        cumulative = np.cumsum(size.values) - size.values
        before_event = np.concatenate(([0], np.cumsum(size.values)))[size.starts[size.event]] if len(size.values) else cumulative
        first_strip = strips.starts[size.event] + (cumulative - before_event)

        sel_size = np.where(valid, size.values[cluster] if len(size.values) else 0, 0).astype(np.int64)
        ok = exclude(sel_size >= 1, f"selected cluster has no strips ({plane})")
        first = np.where(valid, first_strip[cluster] if len(first_strip) else 0, 0)

        def strip_value(offset):
            index = np.clip(first + offset, 0, max(len(strips.values) - 1, 0))
            return strips.values[index] if len(strips.values) else np.full(n, np.nan)

        plane_columns[plane] = {
            "max_strip": strip_value(0),
            "min_strip": strip_value(np.maximum(sel_size - 1, 0)),
            "energy": energy.values[cluster] if len(energy.values) else np.full(n, np.nan),
            "position": position_.values[cluster] if len(position_.values) else np.full(n, np.nan),
            "size": sel_size,
            "texp": texp.values[cluster] if len(texp.values) else np.full(n, np.nan),
            "multiplicity": size.counts,
            "selected": selected,
            "clusters": (size, energy, position_),
        }

    # ---------- assemble the selected events ----------
    reasons["written"] += int(ok.sum())
    if not ok.any():
        return None

    columns = {
        "Phi": S("Phi_4p")[ok].astype(np.float64),
        "Theta": S("Theta_4p")[ok].astype(np.float64),
        "Scattering_xy": out["Scattering_xy"][ok],
        "Scattering_xz": out["Scattering_xz"][ok],
        "ChiSquare_xz": out["ChiSquare_xz"][ok],
        "ChiSquare_xy": out["ChiSquare_xy"][ok],
        "ChiSquare_3p_xz": out["ChiSquare_3p_xz"][ok],
        "ChiSquare_3p_xy": out["ChiSquare_3p_xy"][ok],
    }

    for plane in PLANES:
        columns[f"TrackClusterMaxStripEnergy_{plane}"] = plane_columns[plane]["max_strip"][ok].astype(np.float64)
        columns[f"TrackClusterMinStripEnergy_{plane}"] = plane_columns[plane]["min_strip"][ok].astype(np.float64)
    for key, prefix, dtype in (
        ("energy", "TrackClusterEnergy", np.float64),
        ("position", "TrackClusterPosition", np.float64),
        ("size", "TrackClusterSize", np.int32),
        ("texp", "TrackClusterExp", np.float64),
        ("multiplicity", "ClusterMultriplicity", np.int32),
    ):
        for plane in PLANES:
            columns[f"{prefix}_{plane}"] = plane_columns[plane][key][ok].astype(dtype)

    for plane in PLANES:
        size, energy, position_ = plane_columns[plane]["clusters"]
        local = np.arange(len(size.values)) - size.starts[size.event]
        keep = ok[size.event] & (local != plane_columns[plane]["selected"][size.event])
        counts = np.bincount(size.event[keep], minlength=n)[ok]
        columns[f"NonTrackClusterEnergy_{plane}"] = ak.unflatten(energy.values[keep].astype(np.float64), counts)
        columns[f"NonTrackClusterSize_{plane}"] = ak.unflatten(size.values[keep].astype(np.float64), counts)
        columns[f"NonTrackClusterPosition_{plane}"] = ak.unflatten(position_.values[keep].astype(np.float64), counts)

    columns["Run"] = S("Run")[ok].astype(np.int32)
    columns["TriggerRate"] = S("TriggerRate")[ok].astype(np.float64)
    columns["WorkingPoint"] = S("WorkingPoint")[ok].astype(np.int32)
    columns["timestamp"] = S("timestamp")[ok].astype(np.float64)
    return columns


def branch_types(columns) -> dict:
    types = {}
    for name, values in columns.items():
        if isinstance(values, np.ndarray):
            types[name] = values.dtype
        else:
            types[name] = "var * float64"
    return types


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="AnalyzedData _v1 file")
    parser.add_argument("-o", "--output", type=Path, help="output file (default: <input stem>_golden.root here)")
    parser.add_argument("--max-events", type=int, help="read at most N events, for a quick test")
    parser.add_argument("--chunk", type=int, default=200_000, help="events read at a time (default: 200000)")
    arguments = parser.parse_args()

    output = arguments.output or Path(f"{arguments.input.stem}_golden.root")
    reasons, rule = Counter(), Counter()
    began = time.time()

    with uproot.open(arguments.input) as source:
        if TREE_IN not in source:
            raise SystemExit(f"{arguments.input} has no {TREE_IN}")
        tree = source[TREE_IN]
        total = tree.num_entries if arguments.max_events is None else min(tree.num_entries, arguments.max_events)

        with uproot.recreate(output) as destination:
            created = False

            for start in range(0, total, arguments.chunk):
                stop = min(start + arguments.chunk, total)
                columns = select_chunk(tree, start, stop, reasons, rule)

                if columns is not None:
                    if not created:
                        out_tree = destination.mktree(TREE_OUT, branch_types(columns))
                        created = True
                    out_tree.extend(columns)

                print(f"  {stop:>12,} / {total:,} events", flush=True)

            if not created:
                print("  no events selected; writing an empty tree is skipped")

    written = reasons.pop("written", 0)
    passed = reasons.pop("pass XZ gate", 0)

    print()
    print("==================================================")
    print(f"Golden selection: {arguments.input.name}")
    print("==================================================")
    print(f"  events read          : {total:>12,}")
    print(f"  pass the XZ gate     : {passed:>12,}")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  excluded             : {count:>12,}  {reason}")
    print(f"  written              : {written:>12,}")
    for view in ("xz", "xy"):
        checked = rule[f"{view} checked"]
        agrees = rule[f"{view} agrees"]
        share = f"{agrees / checked:.4%}" if checked else "n/a"
        print(f"  row rule agrees ({view}): {agrees:>12,} / {checked:,}  ({share})")
    print(f"  output               : {output}")
    print(f"  time                 : {time.time() - began:.0f} s")
    print()


if __name__ == "__main__":
    main()
