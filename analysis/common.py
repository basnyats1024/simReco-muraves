"""Shared pieces for the quick-check comparison scripts.

phase1_clusters.py, phase2_tracks.py and phase3_golden.py compare
simulation against data, or one simulation version against another, at
cluster, track and golden-track level. This module holds what they share:

    samples   reading samples.json and resolving file paths
    readers   one reader per output format, each returning the same numpy
              arrays, so every observable is computed by one piece of code
              for every sample
    stats     small summaries
    plots     the lollipop summary figure and the distribution grids
    output    the summary table and a record of what was read

The readers double as examples of how to read each format: simulated
ClusteredData, TrackedData and Tracks4ptree, written as RNTuples, and
experimental AnalyzedData and golden Tracks4ptree, written as TTrees.
uproot reads both kinds identically.

These are quick checks, not a physics analysis. No event selection is
applied, and the number of events behind every quantity is printed, so
the denominators stay visible.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

try:
    from simReco_muraves.tracking.event_builder_adapter import STATION_MAP
except ImportError as error:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "The analysis scripts use the station map from simReco_muraves; "
        "install the package first (pip install -e .)."
    ) from error


# ==========================================================
# Detector layout
# ==========================================================

PLANES = ["Y1", "Y2", "Y3", "Y4", "Z1", "Z2", "Z3", "Z4"]
PLANE_FOR_STATION = {station: plane.upper() for station, plane in STATION_MAP.items()}

#: Planes after the lead absorber; outside the trigger in data.
PLANE_4_INDICES = (3, 7)

#: Tracking planes whose residuals are stored, in output order.
RESIDUAL_PLANES = ["Z1", "Z2", "Z3", "Y1", "Y2", "Y3"]


def view_of(plane: str) -> str:
    """Z planes are measured in the XZ projection, Y planes in XY."""
    return "XZ" if plane.startswith("Z") else "XY"


DIRECTIONS = ("all", "forward", "backward")


def is_forward(phi: np.ndarray) -> np.ndarray:
    """Inside the field of view, 90 < phi < 270 degrees."""
    return (phi > 90) & (phi < 270)


def direction_mask(phi: np.ndarray, direction: str) -> np.ndarray:
    if direction == "all":
        return np.ones(len(phi), dtype=bool)
    forward = is_forward(phi)
    return forward if direction == "forward" else ~forward


# ==========================================================
# Samples
# ==========================================================

DETECTOR_STYLE = {
    "NERO": ("black", "o"),
    "ROSSO": ("#d62728", "s"),
    "BLU": ("#1f5fd6", "^"),
}

SIM_STYLES = [
    ("#2ca02c", "D"),
    ("#17a589", "P"),
    ("#8bc34a", "X"),
    ("#006d2c", "v"),
]

#: Which file entry each kind of sample uses in each phase.
FILE_KEYS = {
    ("sim", 1): "clustered",
    ("sim", 2): "tracked",
    ("sim", 3): "golden",
    ("data", 1): "analyzed",
    ("data", 2): "analyzed",
    ("data", 3): "golden",
}


@dataclass
class Sample:
    name: str
    kind: str
    label: str
    colour: str
    marker: str
    paths: dict = field(default_factory=dict)
    reco_version: str | None = None

    @property
    def display(self) -> str:
        """The label, with the reconstruction version when one is set."""
        return f"{self.label} [{self.reco_version}]" if self.reco_version else self.label


def version_note(samples) -> str:
    """A short note of the data reconstruction versions, for titles."""
    versions = {s.name: s.reco_version for s in samples.values() if s.kind == "data" and s.reco_version}
    if not versions:
        return ""
    distinct = sorted(set(versions.values()))
    if len(distinct) == 1:
        return f"data reconstruction {distinct[0]}"
    return "data reconstruction " + ", ".join(f"{n} {v}" for n, v in versions.items())


def load_samples(config_path: Path, names: list[str] | None = None) -> dict[str, Sample]:
    """Samples from samples.json, in the order requested (default: file order).

    Each entry has a kind ("sim" or "data"), an optional label, colour,
    marker and reco_version, and its files: "clustered", "tracked" and
    "golden" for simulation, "analyzed" and "golden" for data. A file
    entry may be a ROOT file, a directory of ROOT files, a glob pattern,
    or a list of these.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise SystemExit(
            f"No sample configuration at {config_path}. Copy "
            "samples.example.json to samples.json and edit the paths."
        )

    entries = json.loads(config_path.read_text(encoding="utf-8")).get("samples", {})

    if names:
        unknown = [name for name in names if name not in entries]
        if unknown:
            raise SystemExit(f"Unknown samples {unknown}; the configuration has {list(entries)}")
        order = names
    else:
        order = list(entries)

    samples = {}
    sim_count = 0

    for name in order:
        entry = entries[name]
        kind = entry.get("kind")

        if kind not in ("sim", "data"):
            raise SystemExit(f"Sample {name}: kind must be 'sim' or 'data', got {kind!r}")

        if kind == "data":
            detector = str(entry.get("detector", name)).upper()
            colour, marker = DETECTOR_STYLE.get(detector, ("0.45", "o"))
        else:
            colour, marker = SIM_STYLES[sim_count % len(SIM_STYLES)]
            sim_count += 1

        samples[name] = Sample(
            name=name,
            kind=kind,
            label=entry.get("label", name),
            colour=entry.get("colour", colour),
            marker=entry.get("marker", marker),
            paths={key: value for key, value in entry.items() if key in ("clustered", "tracked", "analyzed", "golden")},
            reco_version=entry.get("reco_version"),
        )

    return samples


def resolve_files(spec) -> list[Path]:
    """Expand a file entry into a sorted list of existing ROOT files."""
    specs = spec if isinstance(spec, list) else [spec]
    files: list[Path] = []

    for item in specs:
        path = Path(str(item)).expanduser()

        if path.is_dir():
            files.extend(sorted(path.glob("*.root")))
        elif any(character in str(item) for character in "*?["):
            files.extend(Path(match) for match in sorted(glob.glob(str(path))))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"No such file or directory: {path}")

    if not files:
        raise SystemExit(f"No ROOT files found for {spec}")

    return files


def files_for(sample: Sample, phase: int) -> list[Path]:
    key = FILE_KEYS[(sample.kind, phase)]

    if key not in sample.paths:
        raise SystemExit(f"Sample {sample.name} has no '{key}' entry, needed for phase {phase}")

    return resolve_files(sample.paths[key])


def choose_roles(samples: dict[str, Sample], target: str | None, reference: str | None):
    """The sample under test, and the denominator of the distribution ratios.

    target     numerator of every ratio; default the first simulation
               sample, else the first sample
    reference  denominator of the ratio panels under the distributions;
               default NERO if selected, else the first other sample
    """
    names = list(samples)

    if target is None:
        target = next((name for name in names if samples[name].kind == "sim"), names[0])
    if target not in samples:
        raise SystemExit(f"--target {target} is not among the selected samples {names}")

    others = [name for name in names if name != target]

    if reference is None:
        reference = "NERO" if "NERO" in others else (others[0] if others else None)
    if reference is not None and reference not in others:
        raise SystemExit(f"--reference {reference} must be a selected sample other than the target")

    return target, reference


# ==========================================================
# Low-level reading
# ==========================================================
#
# Every uproot and awkward call lives in this section. Above it and
# below it, everything is plain numpy.


@dataclass
class Jagged:
    """One list per event, stored flat: counts[i] values belong to event i."""

    counts: np.ndarray
    values: np.ndarray

    @property
    def starts(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(self.counts)[:-1])).astype(np.int64)


@dataclass
class Nested:
    """A list of lists per event: rows per event, values per row."""

    row_counts: np.ndarray
    value_counts: np.ndarray
    values: np.ndarray


@contextmanager
def open_tree(path: Path, name: str):
    """The named TTree or RNTuple; uproot treats both the same way."""
    with uproot.open(path) as root_file:
        if name not in root_file:
            raise SystemExit(f"{path} has no {name}; it has {list(root_file.keys())}")
        yield root_file[name]


def entries_to_read(tree, limit: int | None) -> int:
    return tree.num_entries if limit is None else min(tree.num_entries, max(limit, 0))


def read_flat(tree, branch: str, stop: int) -> np.ndarray:
    """A branch with one number per entry."""
    return np.asarray(tree[branch].array(library="np", entry_stop=stop))


def read_jagged(tree, branch: str, stop: int) -> Jagged:
    """A branch with a list of numbers per entry."""
    array = tree[branch].array(library="ak", entry_stop=stop)
    return Jagged(
        counts=np.asarray(ak.to_numpy(ak.num(array, axis=1)), dtype=np.int64),
        values=np.asarray(ak.to_numpy(ak.flatten(array, axis=None))),
    )


def read_nested(tree, branch: str, stop: int) -> Nested:
    """A branch with a list of lists per entry, such as the 4p candidates."""
    array = tree[branch].array(library="ak", entry_stop=stop)
    rows = ak.flatten(array, axis=1)
    return Nested(
        row_counts=np.asarray(ak.to_numpy(ak.num(array, axis=1)), dtype=np.int64),
        value_counts=np.asarray(ak.to_numpy(ak.num(rows, axis=1)), dtype=np.int64),
        values=np.asarray(ak.to_numpy(ak.flatten(array, axis=None))),
    )


def read_provenance_commit(path: Path) -> str | None:
    """The code commit recorded in a simReco output file, if available."""
    try:
        from simReco_muraves.io.provenance_tree import read_chain

        chain = read_chain(path)
    except Exception:
        return None

    if not chain:
        return None

    code = chain[-1].get("code", {})
    return code.get("commit") or code.get("version")


# ==========================================================
# Picking one element per event
# ==========================================================


def pick(jagged: Jagged, index: np.ndarray) -> np.ndarray:
    """jagged[event][index[event]], NaN where the index is out of range."""
    index = np.asarray(index, dtype=np.int64)
    result = np.full(len(jagged.counts), np.nan)
    ok = (index >= 0) & (index < jagged.counts)
    result[ok] = jagged.values[jagged.starts[ok] + index[ok]]
    return result


def pick_nested(nested: Nested, row: np.ndarray, column: np.ndarray) -> np.ndarray:
    """nested[event][row[event]][column[event]], NaN where out of range."""
    row = np.asarray(row, dtype=np.int64)
    column = np.asarray(column, dtype=np.int64)
    result = np.full(len(nested.row_counts), np.nan)

    row_starts = np.concatenate(([0], np.cumsum(nested.row_counts)[:-1])).astype(np.int64)
    value_starts = np.concatenate(([0], np.cumsum(nested.value_counts)[:-1])).astype(np.int64)

    ok = (row >= 0) & (column >= 0) & (row < nested.row_counts)
    events = np.flatnonzero(ok)
    rows = row_starts[events] + row[events]

    in_row = column[events] < nested.value_counts[rows]
    events, rows = events[in_row], rows[in_row]

    result[events] = nested.values[value_starts[rows] + column[events]]
    return result


# ==========================================================
# Cluster-level samples (phase I)
# ==========================================================


@dataclass
class ClusterSample:
    events: int
    counts: dict[str, np.ndarray]        # clusters per event, per plane
    size: dict[str, np.ndarray]          # one value per cluster
    position: dict[str, np.ndarray]
    energy: dict[str, np.ndarray]
    strip_energy: dict[str, np.ndarray]  # one value per strip in a cluster


def read_clusters(sample: Sample, files: list[Path], max_events: int | None) -> ClusterSample:
    if sample.kind == "sim":
        return read_sim_clusters(files, max_events)
    return read_data_clusters(files, max_events)


def read_sim_clusters(files: list[Path], max_events: int | None) -> ClusterSample:
    """Simulated ClusteredData: one entry per cluster.

    Events are identified by (ProcessID, Event). Their clusters are
    gathered across all files, so an event split between two chunks is
    still counted once. Events without any cluster are not in the file.
    """
    columns = {name: [] for name in ("ProcessID", "Event", "Station", "ClusterSize", "ClusterPosition", "ClusterEnergy")}
    strip_counts, strip_values = [], []

    for path in files:
        with open_tree(path, "ClusteredData") as tree:
            stop = tree.num_entries

            for name in columns:
                columns[name].append(read_flat(tree, name, stop))

            strips = read_jagged(tree, "StripEnergies", stop)
            strip_counts.append(strips.counts)
            strip_values.append(strips.values)

        if max_events is not None:
            keys = _event_keys(np.concatenate(columns["ProcessID"]), np.concatenate(columns["Event"]))
            if len(np.unique(keys)) >= max_events:
                break

    data = {name: np.concatenate(parts) for name, parts in columns.items()}
    strips = Jagged(np.concatenate(strip_counts), np.concatenate(strip_values))
    keys = _event_keys(data["ProcessID"], data["Event"])

    if max_events is not None:
        unique, first = np.unique(keys, return_index=True)
        if len(unique) > max_events:
            kept = unique[np.argsort(first)[:max_events]]   # first events in file order
            row_kept = np.isin(keys, kept)
            data = {name: values[row_kept] for name, values in data.items()}
            strips = Jagged(strips.counts[row_kept], strips.values[np.repeat(row_kept, strips.counts)])
            keys = keys[row_kept]

    unique, event_index = np.unique(keys, return_inverse=True)
    n_events = len(unique)
    strip_rows = np.repeat(np.arange(len(strips.counts)), strips.counts)

    sample = ClusterSample(n_events, {}, {}, {}, {}, {})

    for station, plane in PLANE_FOR_STATION.items():
        rows = data["Station"] == station
        sample.counts[plane] = np.bincount(event_index[rows], minlength=n_events)
        sample.size[plane] = data["ClusterSize"][rows]
        sample.position[plane] = data["ClusterPosition"][rows]
        sample.energy[plane] = data["ClusterEnergy"][rows]
        sample.strip_energy[plane] = strips.values[rows[strip_rows]]

    return sample


def read_data_clusters(files: list[Path], max_events: int | None) -> ClusterSample:
    """Experimental AnalyzedData: one entry per triggered event.

    Each plane has its own list branches, e.g. ClusterSize_Y1, and
    StripsEnergy_Y1 holds the strips of that plane's clusters.
    """
    parts = {key: {plane: [] for plane in PLANES} for key in ("counts", "size", "position", "energy", "strips")}
    n_events = 0

    for path in files:
        remaining = None if max_events is None else max_events - n_events
        if remaining is not None and remaining <= 0:
            break

        with open_tree(path, "AnalyzedData") as tree:
            stop = entries_to_read(tree, remaining)

            for plane in PLANES:
                size = read_jagged(tree, f"ClusterSize_{plane}", stop)
                parts["counts"][plane].append(size.counts)
                parts["size"][plane].append(size.values)
                parts["position"][plane].append(read_jagged(tree, f"ClusterPosition_{plane}", stop).values)
                parts["energy"][plane].append(read_jagged(tree, f"ClusterEnergy_{plane}", stop).values)
                parts["strips"][plane].append(read_jagged(tree, f"StripsEnergy_{plane}", stop).values)

        n_events += stop

    def joined(key):
        return {plane: np.concatenate(parts[key][plane]) for plane in PLANES}

    return ClusterSample(
        events=n_events,
        counts=joined("counts"),
        size=joined("size"),
        position=joined("position"),
        energy=joined("energy"),
        strip_energy=joined("strips"),
    )


def _event_keys(process_id: np.ndarray, event: np.ndarray) -> np.ndarray:
    return (process_id.astype(np.int64) << 32) | (event.astype(np.int64) & 0xFFFFFFFF)


# ==========================================================
# Track-level samples (phase II)
# ==========================================================


@dataclass
class TrackSample:
    events: int                        # all events read, whatever the direction
    ntracks: dict[str, np.ndarray]     # 3p tracks per event, per view
    chi2_3p: dict[str, np.ndarray]     # best 3p track, events that have one
    chi2_4p: dict[str, np.ndarray]     # selected 4p track
    theta: dict[str, np.ndarray]       # "3p", "4p": valid tracks only
    phi: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]    # best 3p track, per tracking plane
    scattering: dict[str, np.ndarray]  # selected 4p track, per view
    valid_3p: int
    valid_4p: int


def read_tracks(sample: Sample, files: list[Path], max_events: int | None, direction: str = "all") -> TrackSample:
    reader = _sim_track_file if sample.kind == "sim" else _data_track_file
    tree_name = "TrackedData" if sample.kind == "sim" else "AnalyzedData"
    per_file = []
    n_events = 0

    for path in files:
        remaining = None if max_events is None else max_events - n_events
        if remaining is not None and remaining <= 0:
            break

        with open_tree(path, tree_name) as tree:
            stop = entries_to_read(tree, remaining)
            per_file.append(reader(tree, stop))

        n_events += stop

    raw = {key: np.concatenate([part[key] for part in per_file]) for key in per_file[0]}
    return _derive_tracks(raw, n_events, direction)


def _sim_track_file(tree, stop: int) -> dict[str, np.ndarray]:
    """Simulated TrackedData: the selected 4p track is looked up in the
    nested candidate arrays through its parent-row and inner indices."""
    raw = _common_track_branches(tree, stop)

    for view in ("xz", "xy"):
        parent = read_flat(tree, f"track_3p_of_4p_index_{view}", stop)
        inner = read_flat(tree, f"track_4p_index_{view}", stop)
        raw[f"chi4_{view}"] = pick_nested(read_nested(tree, f"chiSquare_4p_{view}", stop), parent, inner)
        raw[f"scat_{view}"] = pick_nested(read_nested(tree, f"ScatteringAngle_{view}", stop), parent, inner)

    return raw


def _data_track_file(tree, stop: int) -> dict[str, np.ndarray]:
    """Experimental AnalyzedData: the selected 4p track's chi-square and
    scattering angle are stored directly as scalars."""
    raw = _common_track_branches(tree, stop)

    for view in ("xz", "xy"):
        raw[f"chi4_{view}"] = read_flat(tree, f"BestTrack_4p_ChiSquare_{view}", stop).astype(float)
        raw[f"scat_{view}"] = read_flat(tree, f"BestTracks_ScatteringAngle_{view}", stop).astype(float)

    return raw


def _common_track_branches(tree, stop: int) -> dict[str, np.ndarray]:
    """Branches named identically in TrackedData and AnalyzedData."""
    raw = {}

    for view in ("xz", "xy"):
        raw[f"ntracks_{view}"] = read_flat(tree, f"Ntracks_3p_{view}", stop)
        raw[f"chi_{view}"] = read_flat(tree, f"BestChi_{view}", stop).astype(float)
        raw[f"chi_index_{view}"] = read_flat(tree, f"BestChi_{view}_index", stop)

    for kind in ("3p", "4p"):
        raw[f"theta_{kind}"] = read_flat(tree, f"Theta_{kind}", stop).astype(float)
        raw[f"phi_{kind}"] = read_flat(tree, f"Phi_{kind}", stop).astype(float)

    for plane in RESIDUAL_PLANES:
        index = raw["chi_index_xz"] if view_of(plane) == "XZ" else raw["chi_index_xy"]
        raw[f"residual_{plane}"] = pick(read_jagged(tree, f"Residue_Track3p_{plane.lower()}", stop), index)

    return raw


def _derive_tracks(raw: dict[str, np.ndarray], n_events: int, direction: str = "all") -> TrackSample:
    """Per-track quantities, for one direction or all.

    With a direction, 3p quantities keep events whose 3p track points that
    way (phi_3p), 4p quantities those whose 4p track does (phi_4p). Track
    multiplicities stay per event, over all events.
    """
    valid_3p = raw["theta_3p"] >= 0
    valid_4p = raw["theta_4p"] >= 0

    if direction == "all":
        keep_3p = np.ones(n_events, dtype=bool)
        keep_4p = np.ones(n_events, dtype=bool)
    else:
        keep_3p = valid_3p & direction_mask(raw["phi_3p"], direction)
        keep_4p = valid_4p & direction_mask(raw["phi_4p"], direction)

    def finite(values, mask):
        values = values[mask]
        return values[np.isfinite(values)]

    sample = TrackSample(
        events=n_events,
        ntracks={}, chi2_3p={}, chi2_4p={}, theta={}, phi={},
        residual={}, scattering={},
        valid_3p=int((valid_3p & keep_3p).sum()), valid_4p=int((valid_4p & keep_4p).sum()),
    )

    for view in ("xz", "xy"):
        name = view.upper()
        sample.ntracks[name] = raw[f"ntracks_{view}"]
        sample.chi2_3p[name] = finite(raw[f"chi_{view}"], (raw[f"chi_index_{view}"] >= 0) & keep_3p)
        sample.chi2_4p[name] = finite(raw[f"chi4_{view}"], valid_4p & keep_4p)
        sample.scattering[name] = finite(raw[f"scat_{view}"], valid_4p & keep_4p)

    for kind, valid, keep in (("3p", valid_3p, keep_3p), ("4p", valid_4p, keep_4p)):
        sample.theta[kind] = finite(raw[f"theta_{kind}"], valid & keep)
        sample.phi[kind] = finite(raw[f"phi_{kind}"], valid & keep)

    for plane in RESIDUAL_PLANES:
        sample.residual[plane] = finite(raw[f"residual_{plane}"], keep_3p)

    return sample


# ==========================================================
# Golden-track samples (phase III)
# ==========================================================


@dataclass
class GoldenSample:
    """One entry per golden event: its selected four-plane track."""

    events: int
    theta: np.ndarray
    phi: np.ndarray
    scattering: dict[str, np.ndarray]     # "XZ", "XY"
    chi2_3p: dict[str, np.ndarray]
    chi2_4p: dict[str, np.ndarray]
    energy: dict[str, np.ndarray]         # selected cluster, per plane
    position: dict[str, np.ndarray]
    size: dict[str, np.ndarray]
    max_strip: dict[str, np.ndarray]
    multiplicity: dict[str, np.ndarray]   # all clusters on the plane
    nontrack_position: dict[str, Jagged]  # the other clusters on the plane

    def subset(self, mask: np.ndarray) -> "GoldenSample":
        def jagged(j):
            return Jagged(j.counts[mask], j.values[np.repeat(mask, j.counts)])

        per_view = lambda d: {k: v[mask] for k, v in d.items()}
        return GoldenSample(
            events=int(mask.sum()), theta=self.theta[mask], phi=self.phi[mask],
            scattering=per_view(self.scattering), chi2_3p=per_view(self.chi2_3p), chi2_4p=per_view(self.chi2_4p),
            energy=per_view(self.energy), position=per_view(self.position), size=per_view(self.size),
            max_strip=per_view(self.max_strip), multiplicity=per_view(self.multiplicity),
            nontrack_position={p: jagged(j) for p, j in self.nontrack_position.items()},
        )


def read_golden(files: list[Path], max_events: int | None) -> GoldenSample:
    """Simulated or experimental Tracks4ptree.

    The two differ only in spelling: experimental files use the legacy
    ClusterMultriplicity_<plane>, simulated ones ClusterMultiplicity_<plane>.
    """
    parts = []
    n_events = 0

    for path in files:
        remaining = None if max_events is None else max_events - n_events
        if remaining is not None and remaining <= 0:
            break

        with open_tree(path, "Tracks4ptree") as tree:
            stop = entries_to_read(tree, remaining)
            names = set(tree.keys())

            def branch(*options):
                for option in options:
                    if option in names:
                        return option
                raise SystemExit(f"{path} has none of {options}")

            part = {
                "theta": read_flat(tree, "Theta", stop).astype(float),
                "phi": read_flat(tree, "Phi", stop).astype(float),
            }
            for view in ("xz", "xy"):
                part[f"scat_{view}"] = read_flat(tree, f"Scattering_{view}", stop).astype(float)
                part[f"chi3_{view}"] = read_flat(tree, f"ChiSquare_3p_{view}", stop).astype(float)
                part[f"chi4_{view}"] = read_flat(tree, f"ChiSquare_{view}", stop).astype(float)
            for plane in PLANES:
                part[f"energy_{plane}"] = read_flat(tree, f"TrackClusterEnergy_{plane}", stop).astype(float)
                part[f"position_{plane}"] = read_flat(tree, f"TrackClusterPosition_{plane}", stop).astype(float)
                part[f"size_{plane}"] = read_flat(tree, f"TrackClusterSize_{plane}", stop).astype(float)
                part[f"max_{plane}"] = read_flat(tree, f"TrackClusterMaxStripEnergy_{plane}", stop).astype(float)
                part[f"mult_{plane}"] = read_flat(
                    tree, branch(f"ClusterMultiplicity_{plane}", f"ClusterMultriplicity_{plane}"), stop).astype(float)
                nontrack = read_jagged(tree, f"NonTrackClusterPosition_{plane}", stop)
                part[f"ntc_{plane}"] = nontrack.counts
                part[f"ntv_{plane}"] = nontrack.values.astype(float)
            parts.append(part)

        n_events += stop

    joined = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    views = lambda prefix: {v.upper(): joined[f"{prefix}_{v}"] for v in ("xz", "xy")}
    planes = lambda prefix: {p: joined[f"{prefix}_{p}"] for p in PLANES}

    return GoldenSample(
        events=n_events, theta=joined["theta"], phi=joined["phi"],
        scattering=views("scat"), chi2_3p=views("chi3"), chi2_4p=views("chi4"),
        energy=planes("energy"), position=planes("position"), size=planes("size"),
        max_strip=planes("max"), multiplicity=planes("mult"),
        nontrack_position={p: Jagged(joined[f"ntc_{p}"], joined[f"ntv_{p}"]) for p in PLANES},
    )


# ==========================================================
# Statistics
# ==========================================================


def mean(values) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def median(values) -> float:
    return float(np.median(values)) if len(values) else math.nan


def fraction(mask) -> float:
    return float(np.mean(mask)) if len(mask) else math.nan


def rms(values) -> float:
    return float(np.std(values)) if len(values) else math.nan


def width68(values) -> float:
    """Half the 16-84 % range: a width that ignores the far tails."""
    if len(values) < 2:
        return math.nan
    low, high = np.percentile(values, [16, 84])
    return float(high - low) / 2


def ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


# ==========================================================
# Plots
# ==========================================================

RATIO_LIMITS = (0.2, 5.0)
RATIO_TICKS = [0.2, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]

#: Agreement bands as two shades of one hue, so they read as nested; the
#: plane-4 shading in a different hue. None clash with the sample colours.
BAND_5 = "#e3d3a8"
BAND_10 = "#f3ecd6"
PLANE_4_SHADE = "#e8e2f4"

LINE_TARGET = 1.4
LINE_OTHER = 1.0


@dataclass
class SummaryPanel:
    title: str
    categories: list[str]
    values: dict[str, np.ndarray]     # sample -> one value per category
    shaded: tuple = ()                # category indices to shade
    divider: float | None = None      # x position of a dotted divider


@dataclass
class DistributionPanel:
    title: str
    bins: np.ndarray
    values: dict[str, np.ndarray]     # sample -> entries
    xlabel: str = ""
    logy: bool = False
    ratio_log: bool = True
    ratio_limits: tuple = (0.2, 5.0)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def lollipop_summary(panels, samples, target, title, path, ncols=4, shade_label="plane 4"):
    """target ÷ every other sample, one panel per observable.

    Stems run from 1 to each ratio, in the colour of the denominator.
    Ratios beyond the axis are drawn as arrows at its edge, with the value.
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    others = [name for name in samples if name != target]
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.7 * nrows), sharey=True, squeeze=False)
    low, high = RATIO_LIMITS
    spacing = 0.22 if len(others) > 1 else 0.0

    for ax, panel in zip(axes.flat, panels):
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        x = np.arange(len(panel.categories))

        for column in panel.shaded:
            ax.axvspan(column - 0.5, column + 0.5, color=PLANE_4_SHADE, zorder=0)
        if panel.divider is not None:
            ax.axvline(panel.divider, color="0.6", lw=0.8, ls=":")

        ax.axhspan(1 / 1.10, 1.10, color=BAND_10, zorder=0.5)
        ax.axhspan(1 / 1.05, 1.05, color=BAND_5, zorder=0.6)
        ax.axhline(1.0, color="0.3", lw=0.9, zorder=0.7)

        # labelling every off-scale point only stays readable with few categories;
        # with many, the arrows remain and one note points to the summary CSV
        label_points = len(panel.categories) <= 4
        off_scale = False

        for k, other in enumerate(others):
            style = samples[other]
            offset = (k - (len(others) - 1) / 2) * spacing

            for xi, top, bottom in zip(x, panel.values[target], panel.values[other]):
                value = ratio(top, bottom)
                if not np.isfinite(value):
                    continue

                xp = xi + offset
                shown = min(max(value, low * 1.04), high / 1.04)
                ax.plot([xp, xp], [1.0, shown], color=style.colour, lw=1.3, alpha=0.85, zorder=2)

                if low < value < high:
                    ax.plot(xp, shown, marker=style.marker, color=style.colour, ms=5.5,
                            mec="white", mew=0.5, zorder=3)
                else:
                    below = value <= low
                    off_scale = True
                    ax.plot(xp, shown, marker="v" if below else "^", color=style.colour, ms=7, zorder=3)
                    if label_points:
                        ax.annotate(f"{value:.2f}", (xp, shown),
                                    xytext=(0, -(10 + 8 * k) if below else 4 + 8 * k),
                                    textcoords="offset points", ha="center", fontsize=6.5, color=style.colour)

        if off_scale and not label_points:
            ax.text(0.98, 0.97, "▲▼ beyond the axis: values in the summary CSV", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="0.35")

        ax.set_title(panel.title, fontsize=11)
        ax.set_xticks(x, panel.categories, fontsize=9)
        ax.set_xlim(-0.6, len(panel.categories) - 0.4)
        ax.set_yticks(RATIO_TICKS, [f"{t:g}" for t in RATIO_TICKS], fontsize=8)
        ax.minorticks_off()

    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{target} ÷ sample", fontsize=10)

    handles = [
        Line2D([], [], color=samples[o].colour, marker=samples[o].marker, lw=1.3, ms=7, label=f"{target} ÷ {o}")
        for o in others
    ]
    handles += [Patch(color=BAND_5, label="±5 %"), Patch(color=BAND_10, label="±10 %")]
    if any(panel.shaded for panel in panels):
        handles.append(Patch(color=PLANE_4_SHADE, label=shade_label))

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ratio_threshold(n_reference: int, min_count: int | None) -> int:
    """Entries a reference bin needs before its ratio is drawn.

    A fixed --min-count wins; otherwise it scales with the sample, from 10
    entries for small samples up to 100 for large ones, so that panels
    with few entries, such as four-plane quantities, still show a ratio.
    """
    if min_count is not None:
        return min_count
    return int(np.clip(round(n_reference / 2000), 10, 100))


def distribution_grid(panels, samples, events, target, reference, title, path, ncols=4, min_count=None):
    """Each sample's normalised distribution, with target ÷ reference below.

    Distributions are fractions of each sample's entries. The ratio is
    drawn only in bins where the reference has enough entries; see
    ratio_threshold.
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D

    nrows = math.ceil(len(panels) / ncols)
    with_ratio = reference is not None
    legend_columns = min(len(samples), max(2, ncols))
    legend_rows = math.ceil(len(samples) / legend_columns)

    # the title and legend get their own band above the panels, sized for
    # the number of legend rows, so they never overlap the panel titles
    header = 0.75 + 0.36 * legend_rows
    height = (4.0 if with_ratio else 3.1) * nrows + header
    fig = plt.figure(figsize=(4.2 * ncols, height))
    grid = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.28,
                            top=1 - header / height, bottom=0.55 / height)

    for index, panel in enumerate(panels):
        cell = grid[index // ncols, index % ncols]

        if with_ratio:
            inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            ax_ratio = fig.add_subplot(inner[1], sharex=ax)
        else:
            ax, ax_ratio = fig.add_subplot(cell), None

        fractions, counts = {}, {}
        for name, style in samples.items():
            values = panel.values[name]
            counts[name], _ = np.histogram(values, bins=panel.bins)
            fractions[name] = counts[name] / max(len(values), 1)
            ax.stairs(fractions[name], panel.bins, color=style.colour,
                      lw=LINE_TARGET if name == target else LINE_OTHER, zorder=3 if name == target else 2)

        if panel.logy:
            ax.set_yscale("log")
            positive = [f[f > 0].min() for f in fractions.values() if np.any(f > 0)]
            if positive:
                ax.set_ylim(bottom=min(positive) * 0.5)

        ax.set_title(panel.title, fontsize=10.5)
        ax.set_ylabel("fraction", fontsize=8.5)
        ax.tick_params(labelsize=8)

        if ax_ratio is not None:
            plt.setp(ax.get_xticklabels(), visible=False)
            centres = 0.5 * (panel.bins[1:] + panel.bins[:-1])
            shown = counts[reference] >= ratio_threshold(len(panel.values[reference]), min_count)
            values = np.full(len(centres), np.nan)
            ok = shown & (fractions[reference] > 0)
            values[ok] = fractions[target][ok] / fractions[reference][ok]

            ax_ratio.axhspan(0.95, 1.05, color="0.85", zorder=0)
            ax_ratio.axhline(1.0, color="0.3", lw=0.8)
            ax_ratio.plot(centres, values, ls="", marker="o", ms=2.6, color=samples[target].colour)

            if panel.ratio_log:
                ax_ratio.set_yscale("log")
                ticks = _log_ratio_ticks(*panel.ratio_limits)
                ax_ratio.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
            ax_ratio.set_ylim(*panel.ratio_limits)
            ax_ratio.set_ylabel(f"÷ {reference}", fontsize=8)
            ax_ratio.tick_params(labelsize=7.5)
            ax_ratio.minorticks_off()
            ax_ratio.set_xlabel(panel.xlabel, fontsize=8.5)
        else:
            ax.set_xlabel(panel.xlabel, fontsize=8.5)

    handles = [
        Line2D([], [], color=style.colour, lw=LINE_TARGET if name == target else LINE_OTHER,
               label=f"{style.display} ({events[name]:,} events)")
        for name, style in samples.items()
    ]
    fig.suptitle(title, fontsize=12.5, y=1 - 0.12 / height, va="top")
    fig.legend(handles=handles, loc="upper center", ncol=legend_columns, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1 - 0.5 / height))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _log_ratio_ticks(low: float, high: float) -> list[float]:
    """Five readable ticks on a logarithmic ratio axis, centred on 1."""
    def nice(value):
        return float(f"{value:.2g}")

    return [low, nice(math.sqrt(low)), 1.0, nice(math.sqrt(high)), high]


# ==========================================================
# Output
# ==========================================================


@dataclass
class SummaryRow:
    observable: str
    category: str
    values: dict[str, float]
    with_ratio: bool = True


def print_summary(rows: list[SummaryRow], samples, target, title: str) -> None:
    names = list(samples)
    others = [name for name in names if name != target]
    columns = names + [f"{target}/{o}" for o in others]
    width = max(12, max(len(c) for c in columns) + 2)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    observable_width = max(26, max(len(row.observable) for row in rows) + 2)
    category_width = max(6, max(len(row.category) for row in rows) + 2)
    print(f"{'observable':<{observable_width}}{'':<{category_width}}" + "".join(f"{c:>{width}}" for c in columns))

    for row in rows:
        cells = [_fmt(row.values[name]) for name in names]
        cells += [_fmt(ratio(row.values[target], row.values[o])) if row.with_ratio else "" for o in others]
        print(f"{row.observable:<{observable_width}}{row.category:<{category_width}}"
              + "".join(f"{cell:>{width}}" for cell in cells))


def write_summary(rows: list[SummaryRow], samples, target, path: Path) -> None:
    names = list(samples)
    others = [name for name in names if name != target]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["observable", "category"] + names + [f"{target}/{o}" for o in others])

        for row in rows:
            ratios = [ratio(row.values[target], row.values[o]) if row.with_ratio else "" for o in others]
            writer.writerow([row.observable, row.category] + [row.values[n] for n in names] + ratios)


def write_run_info(path: Path, phase: int, samples, files, events, target, reference, max_events,
                   direction: str = "all") -> None:
    info = {
        "phase": phase,
        "target": target,
        "reference": reference,
        "max_events": max_events,
        "direction": direction,
        "samples": {
            name: {
                "kind": sample.kind,
                "label": sample.label,
                "reco_version": sample.reco_version,
                "events": events[name],
                "files": [str(f) for f in files[name]],
                "code": read_provenance_commit(files[name][0]) if sample.kind == "sim" else None,
            }
            for name, sample in samples.items()
        },
    }
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.4g}"


# ==========================================================
# Command line
# ==========================================================


def argument_parser(description: str, default_out: str, with_direction: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("samples.json"),
                        help="sample configuration (default: samples.json next to this script)")
    parser.add_argument("--samples", nargs="+", metavar="NAME",
                        help="samples to compare, in plotting order (default: all in the configuration)")
    parser.add_argument("--target", metavar="NAME",
                        help="sample under test, numerator of every ratio (default: first simulation sample)")
    parser.add_argument("--reference", metavar="NAME",
                        help="denominator of the ratio panels (default: NERO, else the first other sample)")
    parser.add_argument("--max-events", type=int, metavar="N",
                        help="read at most N events per sample, for quick runs")
    parser.add_argument("--out", type=Path, default=Path(default_out),
                        help=f"output directory (default: {default_out})")
    parser.add_argument("--min-count", type=int, metavar="N",
                        help="show a ratio only where the reference bin has N entries "
                             "(default: scales with the sample, 10 to 100)")
    if with_direction:
        parser.add_argument("--direction", choices=DIRECTIONS, default="all",
                            help="restrict track quantities to forward (90 < phi < 270) or backward tracks "
                                 "(default: all)")
    return parser


def describe_samples(samples, files, events, target, reference) -> None:
    print()
    print("Samples")
    print("--------------------------------------------------")
    for name, sample in samples.items():
        role = " (target)" if name == target else (" (reference)" if name == reference else "")
        version = f"  [{sample.reco_version}]" if sample.reco_version else ""
        print(f"  {name:<10} {sample.kind:<5} {events[name]:>12,} events  {len(files[name])} file(s){version}{role}")
