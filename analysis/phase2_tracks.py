#!/usr/bin/env python3
"""Phase II quick check: track-level comparison.

Compares the tracking output: how many three-plane tracks each event
has, how well the best ones fit, their angles and residuals, and the
four-plane extension's fit, scattering angle and acceptance. Simulation
is read from TrackedData, data from AnalyzedData.

    python analysis/phase2_tracks.py
    python analysis/phase2_tracks.py --samples SIM NERO BLU --max-events 50000
    python analysis/phase2_tracks.py --samples SIM_old SIM --target SIM

Writes to --out:

    phase2_summary.png     target ÷ every other sample, per observable
    phase2_summary.csv     the same numbers, with each sample's values
    phase2_tracks.png      3p tracks per event, best 3p and selected 4p chi-square
    phase2_angles.png      theta and phi of the 3p and 4p tracks
    phase2_residuals.png   best-3p-track residuals per tracking plane
    phase2_scattering.png  scattering angle of the selected 4p track
    phase2_run_info.json   samples, files, events and code version read

Residuals and chi-square are taken from the best three-plane track of
each view; the four-plane quantities from the selected four-plane track.
Cluster timing is left out: the simulated cluster time and the data's
Texp are not the same quantity.
"""

from __future__ import annotations

import numpy as np

import common as c


def forward_fraction(phi: np.ndarray) -> float:
    """Fraction of tracks inside the field of view, 90 < phi < 270 degrees."""
    return c.fraction((phi > 90) & (phi < 270))


#: (title, categories, function of a TrackSample returning one value per category, divider)
SUMMARY = [
    ("3p tracks / event", ["XZ", "XY"],
     lambda s: [c.mean(s.ntracks[v]) for v in ("XZ", "XY")], None),
    ("Events with a 3p track", ["XZ", "XY"],
     lambda s: [c.fraction(s.ntracks[v] >= 1) for v in ("XZ", "XY")], None),
    ("Best 3p χ², median", ["XZ", "XY"],
     lambda s: [c.median(s.chi2_3p[v]) for v in ("XZ", "XY")], None),
    ("Selected 4p χ², median", ["XZ", "XY"],
     lambda s: [c.median(s.chi2_4p[v]) for v in ("XZ", "XY")], None),
    ("θ, median", ["3p", "4p"],
     lambda s: [c.median(s.theta[k]) for k in ("3p", "4p")], None),
    ("φ inside field of view", ["3p", "4p"],
     lambda s: [forward_fraction(s.phi[k]) for k in ("3p", "4p")], None),
    ("Residual width (68 %)", c.RESIDUAL_PLANES,
     lambda s: [c.width68(s.residual[p]) for p in c.RESIDUAL_PLANES], 2.5),
    ("Scattering width (68 %)", ["XZ", "XY"],
     lambda s: [c.width68(s.scattering[v]) for v in ("XZ", "XY")], None),
    ("Four-plane acceptance", ["4p ÷ 3p"],
     lambda s: [c.ratio(s.valid_4p, s.valid_3p)], None),
]


def residual_bins(plane: str) -> np.ndarray:
    """The middle plane's residuals are about twice as wide as the outer ones'."""
    half_width = 0.013 if plane.endswith("2") else 0.0065
    return np.linspace(-half_width, half_width, 66)


def distribution_figures(tracks):
    """(file key, title, columns, panels) for each distribution figure."""

    def panel(title, entries, bins, xlabel, logy=False, ratio_log=True, ratio_limits=(0.2, 5.0)):
        return c.DistributionPanel(
            title, bins, {name: entries(data) for name, data in tracks.items()},
            xlabel=xlabel, logy=logy, ratio_log=ratio_log, ratio_limits=ratio_limits,
        )

    track_panels = []
    for view in ("XZ", "XY"):
        track_panels += [
            panel(f"3p tracks per event, {view}", lambda s, v=view: s.ntracks[v],
                  np.arange(-0.5, 7.5), "tracks", logy=True),
            panel(f"Best 3p χ², {view}", lambda s, v=view: s.chi2_3p[v],
                  np.linspace(0, 15, 61), "χ²", logy=True),
            panel(f"Selected 4p χ², {view}", lambda s, v=view: s.chi2_4p[v],
                  np.linspace(0, 50, 51), "χ²", logy=True),
        ]

    angle_panels = [
        panel("θ, 3p", lambda s: s.theta["3p"], np.linspace(0, 70, 71), "θ [deg]", ratio_limits=(0.5, 2.0)),
        panel("φ, 3p", lambda s: s.phi["3p"], np.linspace(0, 360, 73), "φ [deg]", ratio_limits=(0.5, 2.0)),
        panel("θ, 4p", lambda s: s.theta["4p"], np.linspace(0, 40, 41), "θ [deg]", ratio_limits=(0.5, 2.0)),
        panel("φ, 4p", lambda s: s.phi["4p"], np.linspace(0, 360, 73), "φ [deg]", ratio_limits=(0.5, 2.0)),
    ]

    residual_panels = [
        panel(f"Residual, {plane}", lambda s, p=plane: s.residual[p], residual_bins(plane), "residual [m]")
        for plane in c.RESIDUAL_PLANES
    ]

    scattering_panels = [
        panel(f"Scattering angle, {view}", lambda s, v=view: s.scattering[v],
              np.linspace(-25, 25, 51), "angle [deg]", logy=True, ratio_limits=(0.1, 10.0))
        for view in ("XZ", "XY")
    ]

    return [
        ("tracks", "tracks per event and fit quality", 3, track_panels),
        ("angles", "track angles", 2, angle_panels),
        ("residuals", "best-track residuals", 3, residual_panels),
        ("scattering", "four-plane scattering angle", 2, scattering_panels),
    ]


def main() -> None:
    parser = c.argument_parser(__doc__, "analysis_output/phase2")
    arguments = parser.parse_args()

    samples = c.load_samples(arguments.config, arguments.samples)
    target, reference = c.choose_roles(samples, arguments.target, arguments.reference)
    arguments.out.mkdir(parents=True, exist_ok=True)

    files, tracks = {}, {}
    for name, sample in samples.items():
        files[name] = c.files_for(sample, phase=2)
        print(f"reading {name} ({len(files[name])} file(s)) ...", flush=True)
        tracks[name] = c.read_tracks(sample, files[name], arguments.max_events)

    events = {name: data.events for name, data in tracks.items()}
    c.describe_samples(samples, files, events, target, reference)

    # ---------- summary ----------
    rows = [
        c.SummaryRow("Events", "all", dict(events), with_ratio=False),
        c.SummaryRow("Valid 3p tracks", "all", {n: d.valid_3p for n, d in tracks.items()}, with_ratio=False),
        c.SummaryRow("Valid 4p tracks", "all", {n: d.valid_4p for n, d in tracks.items()}, with_ratio=False),
    ]
    panels = []

    for title, categories, function, divider in SUMMARY:
        values = {name: np.array(function(data), dtype=float) for name, data in tracks.items()}
        panels.append(c.SummaryPanel(title, categories, values, divider=divider))
        rows += [
            c.SummaryRow(title, category, {name: float(values[name][i]) for name in samples})
            for i, category in enumerate(categories)
        ]

    c.print_summary(rows, samples, target, "Phase II — track-level summary")
    c.write_summary(rows, samples, target, arguments.out / "phase2_summary.csv")

    if len(samples) > 1:
        c.lollipop_summary(
            panels, samples, target,
            f"Phase II — track-level summary, {target} ÷ sample",
            arguments.out / "phase2_summary.png",
            ncols=3,
        )

    # ---------- distributions ----------
    for key, title, columns, figure_panels in distribution_figures(tracks):
        c.distribution_grid(
            figure_panels, samples, events, target, reference,
            f"Phase II — {title}", arguments.out / f"phase2_{key}.png",
            ncols=columns, min_count=arguments.min_count,
        )

    c.write_run_info(arguments.out / "phase2_run_info.json", 2, samples, files, events,
                     target, reference, arguments.max_events)

    print()
    print(f"Written to {arguments.out}/")


if __name__ == "__main__":
    main()
