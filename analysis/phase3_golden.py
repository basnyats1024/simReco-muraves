#!/usr/bin/env python3
"""Phase III quick check: golden-track comparison.

Compares the golden selection's output, one selected four-plane track per
event, between samples: its angles, scattering and fit quality; the
clusters it uses on each plane; and the other activity on those planes.
Simulation and data are both read from Tracks4ptree, simulated files
spelling ClusterMultiplicity_<plane> and experimental ones the legacy
ClusterMultriplicity_<plane>.

    python analysis/phase3_golden.py
    python analysis/phase3_golden.py --direction forward --out analysis_output/phase3_forward
    python analysis/phase3_golden.py --samples SIM_old SIM --target SIM

Writes to --out:

    phase3_summary.png          target ÷ every other sample, per observable
    phase3_summary.csv          the same numbers, with each sample's values
    phase3_position_medians.csv selected-cluster position medians, and the
                                target's shift from each sample in cm
    phase3_plane4_table.md/.tex plane 4, all clusters (Phase I) against
                                clusters on the track (here), for slides
    phase3_tracks.png           theta, phi, scattering, chi-square
    phase3_position.png         selected-cluster position
    phase3_energy.png           selected-cluster energy
    phase3_size.png             selected-cluster size
    phase3_leading_strip.png    leading-strip energy ÷ cluster energy
    phase3_multiplicity.png     clusters per plane
    phase3_nontrack_distance.png  distance of other clusters from the track's
    phase3_run_info.json        samples, files, events and versions read

Direction. In the real setup Mt. Vesuvius lies behind the telescope and
absorbs low-elevation backward muons; the simulation has no mountain. The
summary therefore always shows theta for forward, backward and all
tracks, and the forward/backward ratio, from every golden event.
--direction forward|backward restricts every other quantity to one
direction (90 < phi < 270 is forward). Position medians in particular
shift with the forward/backward mixture.

Cluster timing is left out: the simulated cluster time and the data's
Texp are not the same quantity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import common as c


# ==========================================================
# Observables
# ==========================================================


def leading_fraction(g, plane):
    with np.errstate(divide="ignore", invalid="ignore"):
        values = g.max_strip[plane] / g.energy[plane]
    return values[np.isfinite(values)]


def nontrack_distance(g, plane):
    """|position of each other cluster - position of the track's cluster| [m]."""
    other = g.nontrack_position[plane]
    return np.abs(other.values - np.repeat(g.position[plane], other.counts))


def tail(values, limit):
    return c.fraction(np.abs(values) > limit)


def summary_panels(full, selected, near):
    """(title, categories, function of (full, selected) -> values, shaded, divider)."""
    planes = c.PLANES

    def per_plane(function):
        return lambda f, s: [function(s, p) for p in planes]

    return [
        ("θ median, by direction", ["forward", "backward", "all"],
         lambda f, s: [c.median(f.theta[c.is_forward(f.phi)]), c.median(f.theta[~c.is_forward(f.phi)]),
                       c.median(f.theta)], (), None),
        ("Forward ÷ backward", ["golden"],
         lambda f, s: [c.ratio(float(np.sum(c.is_forward(f.phi))), float(np.sum(~c.is_forward(f.phi))))], (), None),
        ("Scattering width (68 %)", ["XZ", "XY"],
         lambda f, s: [c.width68(s.scattering[v]) for v in ("XZ", "XY")], (), None),
        ("Scattering tail, |angle| > 5°", ["XZ", "XY"],
         lambda f, s: [tail(s.scattering[v], 5.0) for v in ("XZ", "XY")], (), None),
        ("χ², median", ["3p XZ", "3p XY", "4p XZ", "4p XY"],
         lambda f, s: [c.median(s.chi2_3p["XZ"]), c.median(s.chi2_3p["XY"]),
                       c.median(s.chi2_4p["XZ"]), c.median(s.chi2_4p["XY"])], (), 1.5),
        ("4p χ² tail, > 10", ["XZ", "XY"],
         lambda f, s: [c.fraction(s.chi2_4p[v] > 10) for v in ("XZ", "XY")], (), None),
        ("Position RMS", planes, per_plane(lambda g, p: c.rms(g.position[p])), c.PLANE_4_INDICES, 3.5),
        ("Median cluster energy", planes, per_plane(lambda g, p: c.median(g.energy[p])), c.PLANE_4_INDICES, 3.5),
        ("Single-strip fraction", planes, per_plane(lambda g, p: c.fraction(g.size[p] == 1)), c.PLANE_4_INDICES, 3.5),
        ("Leading-strip fraction, median", planes, per_plane(lambda g, p: c.median(leading_fraction(g, p))),
         c.PLANE_4_INDICES, 3.5),
        ("Events with extra clusters", planes, per_plane(lambda g, p: c.fraction(g.multiplicity[p] > 1)),
         c.PLANE_4_INDICES, 3.5),
        (f"Other clusters within {near * 100:g} cm", planes,
         per_plane(lambda g, p: c.fraction(nontrack_distance(g, p) < near)), c.PLANE_4_INDICES, 3.5),
    ]


def distribution_figures(selected):
    def panel(title, entries, bins, xlabel, logy=False, ratio_limits=(0.2, 5.0)):
        return c.DistributionPanel(title, bins, {n: entries(g) for n, g in selected.items()},
                                   xlabel=xlabel, logy=logy, ratio_limits=ratio_limits)

    def per_plane(entries, bins, xlabel, logy=False, ratio_limits=(0.2, 5.0)):
        return [panel(p, lambda g, p=p: entries(g, p), bins, xlabel, logy, ratio_limits) for p in c.PLANES]

    track_panels = [
        panel("θ", lambda g: g.theta, np.linspace(0, 40, 41), "θ [deg]", ratio_limits=(0.5, 2.0)),
        panel("φ", lambda g: g.phi, np.linspace(0, 360, 73), "φ [deg]", logy=True),
        panel("Scattering angle, XZ", lambda g: g.scattering["XZ"], np.linspace(-25, 25, 51), "angle [deg]",
              logy=True, ratio_limits=(0.1, 10.0)),
        panel("Scattering angle, XY", lambda g: g.scattering["XY"], np.linspace(-25, 25, 51), "angle [deg]",
              logy=True, ratio_limits=(0.1, 10.0)),
        panel("3p χ², XZ", lambda g: g.chi2_3p["XZ"], np.linspace(0, 5, 51), "χ²", logy=True),
        panel("3p χ², XY", lambda g: g.chi2_3p["XY"], np.linspace(0, 5, 51), "χ²", logy=True),
        panel("4p χ², XZ", lambda g: g.chi2_4p["XZ"], np.linspace(0, 50, 51), "χ²", logy=True),
        panel("4p χ², XY", lambda g: g.chi2_4p["XY"], np.linspace(0, 50, 51), "χ²", logy=True),
    ]

    return [
        ("tracks", "selected track", track_panels),
        ("position", "selected-cluster position",
         per_plane(lambda g, p: g.position[p], np.linspace(-0.55, 0.55, 45), "position [m]", ratio_limits=(0.5, 2.0))),
        ("energy", "selected-cluster energy",
         per_plane(lambda g, p: g.energy[p], np.linspace(0, 150, 76), "energy [nPE]", True, (0.1, 10.0))),
        ("size", "selected-cluster size",
         per_plane(lambda g, p: g.size[p], np.arange(0.5, 10.5), "strips", True)),
        ("leading_strip", "leading-strip fraction",
         per_plane(leading_fraction, np.linspace(0, 1.0, 41), "max strip ÷ cluster energy", False, (0.5, 2.0))),
        ("multiplicity", "clusters per plane",
         per_plane(lambda g, p: g.multiplicity[p], np.arange(-0.5, 7.5), "clusters", True)),
        ("nontrack_distance", "distance of other clusters from the track's",
         per_plane(nontrack_distance, np.linspace(0, 1.0, 41), "|Δ position| [m]", True)),
    ]


# ==========================================================
# Tables
# ==========================================================


def position_medians(selected, target, path: Path) -> list[str]:
    others = [n for n in selected if n != target]
    header = ["plane"] + [f"{n} [m]" for n in selected] + [f"{target} - {o} [cm]" for o in others]
    lines = []

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for plane in c.PLANES:
            medians = {n: c.median(g.position[plane]) for n, g in selected.items()}
            shifts = [100 * (medians[target] - medians[o]) for o in others]
            writer.writerow([plane] + [f"{medians[n]:.4f}" for n in selected] + [f"{s:+.2f}" for s in shifts])
            lines.append(f"  {plane:<6}" + "".join(f"{medians[n]:>10.4f}" for n in selected)
                         + "".join(f"{s:>+12.2f}" for s in shifts))

    title = f"  {'plane':<6}" + "".join(f"{n:>10}" for n in selected) + "".join(f"{target + '-' + o:>12}" for o in others)
    return [title] + lines


def plane4_table(selected, samples, target, phase1_csv: Path, direction: str, out: Path) -> str:
    """Markdown and LaTeX: target ÷ each data sample, all clusters against on-track clusters."""
    detectors = [n for n in selected if n != target and samples[n].kind == "data"] or \
                [n for n in selected if n != target]
    observables = [("Median cluster energy", lambda g, p: c.median(g.energy[p])),
                   ("Single-strip fraction", lambda g, p: c.fraction(g.size[p] == 1))]
    reference_planes = ["Y1", "Y2", "Y3", "Z1", "Z2", "Z3"]

    phase1 = {}
    if phase1_csv.is_file():
        with open(phase1_csv, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                phase1[(row["observable"], row["category"])] = row

    def all_clusters(observable, plane, detector):
        row = phase1.get((observable, plane))
        try:
            return float(row[target]) / float(row[detector])
        except (TypeError, KeyError, ValueError, ZeroDivisionError):
            return np.nan

    def on_track(function, plane, detector):
        return c.ratio(function(selected[target], plane), function(selected[detector], plane))

    rows = []
    for name, function in observables:
        for label, planes in (("Y4", ["Y4"]), ("Z4", ["Z4"]), ("planes 1–3, mean", reference_planes)):
            cells = []
            for detector in detectors:
                cells.append(np.nanmean([all_clusters(name, p, detector) for p in planes]) if phase1 else np.nan)
                cells.append(np.nanmean([on_track(function, p, detector) for p in planes]))
            rows.append((name, label, cells))

    fmt = lambda v: f"{v:.2f}" if np.isfinite(v) else "–"
    scope = "all golden events" if direction == "all" else f"{direction} golden tracks"
    source = f"all clusters from {phase1_csv}" if phase1 else "Phase I summary not found: all-cluster columns empty"
    note = c.version_note(samples)

    md = [f"**{target} ÷ data, plane 4 against planes 1–3** — all clusters (Phase I) against clusters on the "
          f"golden track ({scope}){'; ' + note if note else ''}.", ""]
    md.append("| Observable | Plane | " + " | ".join(f"{d}: all clusters | {d}: on track" for d in detectors) + " |")
    md.append("|---|---|" + "---:|" * (2 * len(detectors)))
    previous = None
    for name, label, cells in rows:
        md.append(f"| {name if name != previous else ''} | {label} | " + " | ".join(fmt(v) for v in cells) + " |")
        previous = name
    md += ["", f"_{source}_"]
    (out / "phase3_plane4_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    columns = "ll" + "rr" * len(detectors)
    tex = [
        "% requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{target} $\\div$ data on plane 4 and planes 1--3: all clusters against clusters on the "
        f"golden track ({scope}){'; ' + note if note else ''}.}}",
        f"  \\begin{{tabular}}{{{columns}}}",
        "    \\toprule",
        "     & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in detectors) + " \\\\",
        "    " + "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(detectors))),
        "    Observable & Plane & " + " & ".join("all & on track" for _ in detectors) + " \\\\",
        "    \\midrule",
    ]
    previous = None
    for name, label, cells in rows:
        if previous is not None and name != previous:
            tex.append("    \\midrule")
        tex.append(f"    {name if name != previous else ''} & {label.replace('–', '--')} & "
                   + " & ".join(fmt(v).replace("–", "--") for v in cells) + " \\\\")
        previous = name
    tex += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (out / "phase3_plane4_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    return "\n".join(md)


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    parser = c.argument_parser(__doc__, "analysis_output/phase3", with_direction=True)
    parser.add_argument("--phase1-summary", type=Path, default=Path("analysis_output/phase1/phase1_summary.csv"),
                        help="Phase I summary, for the all-cluster columns of the plane-4 table")
    parser.add_argument("--near", type=float, default=0.05, metavar="METRES",
                        help="'near the track' for other clusters (default: 0.05)")
    arguments = parser.parse_args()

    samples = c.load_samples(arguments.config, arguments.samples)
    target, reference = c.choose_roles(samples, arguments.target, arguments.reference)
    arguments.out.mkdir(parents=True, exist_ok=True)

    files, full, selected = {}, {}, {}
    for name, sample in samples.items():
        files[name] = c.files_for(sample, phase=3)
        print(f"reading {name} ({len(files[name])} file(s)) ...", flush=True)
        full[name] = c.read_golden(files[name], arguments.max_events)
        selected[name] = full[name].subset(c.direction_mask(full[name].phi, arguments.direction))

    events = {name: g.events for name, g in selected.items()}
    c.describe_samples(samples, files, {n: g.events for n, g in full.items()}, target, reference)

    notes = [n for n in (c.version_note(samples),
                         "" if arguments.direction == "all" else f"{arguments.direction} tracks") if n]
    suffix = f"  ({'; '.join(notes)})" if notes else ""

    # ---------- summary ----------
    rows = [
        c.SummaryRow("Golden events", "all", {n: g.events for n, g in full.items()}, with_ratio=False),
        c.SummaryRow("Golden events used", arguments.direction, dict(events), with_ratio=False),
    ]
    panels = []
    for title, categories, function, shaded, divider in summary_panels(full, selected, arguments.near):
        values = {n: np.array(function(full[n], selected[n]), dtype=float) for n in samples}
        panels.append(c.SummaryPanel(title, categories, values, shaded=shaded, divider=divider))
        rows += [c.SummaryRow(title, category, {n: float(values[n][i]) for n in samples})
                 for i, category in enumerate(categories)]

    c.print_summary(rows, samples, target, f"Phase III — golden-track summary{suffix}")
    c.write_summary(rows, samples, target, arguments.out / "phase3_summary.csv")

    if len(samples) > 1:
        c.lollipop_summary(panels, samples, target, f"Phase III — golden-track summary, {target} ÷ sample{suffix}",
                           arguments.out / "phase3_summary.png", ncols=4,
                           shade_label="plane 4 (not in the data trigger)")

        print()
        print("Selected-cluster position medians")
        print("--------------------------------------------------")
        print("\n".join(position_medians(selected, target, arguments.out / "phase3_position_medians.csv")))

        print()
        print(plane4_table(selected, samples, target, arguments.phase1_summary, arguments.direction, arguments.out))

    # ---------- distributions ----------
    for key, title, figure_panels in distribution_figures(selected):
        c.distribution_grid(figure_panels, samples, events, target, reference,
                            f"Phase III — {title}{suffix}", arguments.out / f"phase3_{key}.png",
                            ncols=4, min_count=arguments.min_count)

    c.write_run_info(arguments.out / "phase3_run_info.json", 3, samples, files, events,
                     target, reference, arguments.max_events, arguments.direction)

    print()
    print(f"Written to {arguments.out}/")


if __name__ == "__main__":
    main()
