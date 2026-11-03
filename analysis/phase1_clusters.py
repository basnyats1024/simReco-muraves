#!/usr/bin/env python3
"""Phase I quick check: cluster-level comparison.

Compares samples plane by plane: how many clusters each event has, how
large they are, where they sit, and how much energy they and their strips
carry. Simulation is read from ClusteredData, so a digitization or
clustering change can be checked straight after clustering, without
running tracking.

    python analysis/phase1_clusters.py
    python analysis/phase1_clusters.py --samples SIM NERO --max-events 50000
    python analysis/phase1_clusters.py --samples SIM_old SIM --target SIM

Writes to --out:

    phase1_summary.png        target ÷ every other sample, per observable and plane
    phase1_summary.csv        the same numbers, with each sample's values
    phase1_multiplicity.png   clusters per event
    phase1_size.png           cluster size
    phase1_position.png       cluster position
    phase1_cluster_energy.png cluster energy
    phase1_strip_energy.png   energy of the strips in clusters
    phase1_run_info.json      samples, files, events and code version read

Plane 4 sits behind the lead and is not part of the data trigger, so its
numbers are compared as normal but shaded in the summary figure.
"""

from __future__ import annotations

import numpy as np

import common as c


#: (title, function of a ClusterSample and a plane)
SUMMARY = [
    ("Occupancy",              lambda s, p: c.fraction(s.counts[p] > 0)),
    ("Clusters / event",       lambda s, p: c.mean(s.counts[p])),
    ("Multi-cluster fraction", lambda s, p: c.fraction(s.counts[p] > 1)),
    ("Mean cluster size",      lambda s, p: c.mean(s.size[p])),
    ("Single-strip fraction",  lambda s, p: c.fraction(s.size[p] == 1)),
    ("Position RMS",           lambda s, p: c.rms(s.position[p])),
    ("Median cluster energy",  lambda s, p: c.median(s.energy[p])),
    ("Median strip energy",    lambda s, p: c.median(s.strip_energy[p])),
]

#: (file key, title, entries, bins, x label, log y, ratio log, ratio limits)
DISTRIBUTIONS = [
    ("multiplicity", "Clusters per event", lambda s, p: s.counts[p],
     np.arange(-0.5, 7.5), "clusters / event", True, True, (0.2, 5.0)),
    ("size", "Cluster size", lambda s, p: s.size[p],
     np.arange(0.5, 10.5), "strips", True, True, (0.2, 5.0)),
    ("position", "Cluster position", lambda s, p: s.position[p],
     np.linspace(-0.55, 0.55, 45), "position [m]", False, False, (0.75, 1.25)),
    ("cluster_energy", "Cluster energy", lambda s, p: s.energy[p],
     np.linspace(0, 150, 76), "energy [nPE]", True, True, (0.1, 10.0)),
    ("strip_energy", "Strip energy", lambda s, p: s.strip_energy[p],
     np.linspace(0, 150, 76), "energy [nPE]", True, True, (0.1, 10.0)),
]


def main() -> None:
    parser = c.argument_parser(__doc__, "analysis_output/phase1")
    arguments = parser.parse_args()

    samples = c.load_samples(arguments.config, arguments.samples)
    target, reference = c.choose_roles(samples, arguments.target, arguments.reference)
    arguments.out.mkdir(parents=True, exist_ok=True)

    files, clusters = {}, {}
    for name, sample in samples.items():
        files[name] = c.files_for(sample, phase=1)
        print(f"reading {name} ({len(files[name])} file(s)) ...", flush=True)
        clusters[name] = c.read_clusters(sample, files[name], arguments.max_events)

    events = {name: data.events for name, data in clusters.items()}
    c.describe_samples(samples, files, events, target, reference)

    # ---------- summary ----------
    rows = [c.SummaryRow("Events", "all", dict(events), with_ratio=False)]
    panels = []

    for title, function in SUMMARY:
        values = {name: np.array([function(data, plane) for plane in c.PLANES]) for name, data in clusters.items()}
        panels.append(c.SummaryPanel(title, c.PLANES, values, shaded=c.PLANE_4_INDICES, divider=3.5))
        rows += [
            c.SummaryRow(title, plane, {name: float(values[name][i]) for name in samples})
            for i, plane in enumerate(c.PLANES)
        ]

    note = c.version_note(samples)
    suffix = f"  ({note})" if note else ""
    c.print_summary(rows, samples, target, f"Phase I — cluster-level summary{suffix}")
    c.write_summary(rows, samples, target, arguments.out / "phase1_summary.csv")

    if len(samples) > 1:
        c.lollipop_summary(
            panels, samples, target,
            f"Phase I — cluster-level summary, {target} ÷ sample{suffix}",
            arguments.out / "phase1_summary.png",
            shade_label="plane 4 (not in the data trigger)",
        )

    # ---------- distributions ----------
    for key, title, entries, bins, xlabel, logy, ratio_log, ratio_limits in DISTRIBUTIONS:
        panels = [
            c.DistributionPanel(
                plane, bins, {name: entries(data, plane) for name, data in clusters.items()},
                xlabel=xlabel, logy=logy, ratio_log=ratio_log, ratio_limits=ratio_limits,
            )
            for plane in c.PLANES
        ]
        c.distribution_grid(
            panels, samples, events, target, reference,
            f"Phase I — {title.lower()}{suffix}", arguments.out / f"phase1_{key}.png",
            ncols=4, min_count=arguments.min_count,
        )

    c.write_run_info(arguments.out / "phase1_run_info.json", 1, samples, files, events,
                     target, reference, arguments.max_events)

    print()
    print(f"Written to {arguments.out}/")


if __name__ == "__main__":
    main()
