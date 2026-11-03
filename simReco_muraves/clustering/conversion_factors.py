# simReco_muraves/clustering/conversion_factors.py

"""Conversion factors (nPE per MeV) for the digitizer.

The digitizer takes either one factor for every plane, or a mapping
from station ID to factor. This module turns the command-line options
into one or the other, enforces the rules between the options, and
checks the result.

The one piece still missing is resolve_conversion_factors(): what the
factors file looks like, and how a detector and run range select a
calibration from it, is still being designed. It is a stub. Everything
around it -- the flags and their rules, the digitizer, provenance and
output naming -- is in place, so implementing that single function is
all that is needed to enable --pe-per-mev-file.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
from collections import Counter

from simReco_muraves.tracking.event_builder_adapter import STATION_MAP

DETECTORS = ("NERO", "ROSSO", "BLU")

PLANES = ("Y1", "Y2", "Y3", "Y4", "Z1", "Z2", "Z3", "Z4")

NPECM_VALID = (5.0, 15.0) #Based on current fit range, might change later

FALLBACK_SETTING = {
    "NERO": "wp25_a",
    "BLU": "wp25_a",
    "ROSSO": "wp25_b",
}

RunRange = tuple[int, int]


# ----------------------------------------------------------------------
# Run ranges
# ----------------------------------------------------------------------

def parse_run_range(text: str) -> RunRange:
    """Parse '5636-7056', or '5636' meaning that single run."""
    parts = [part.strip() for part in str(text).strip().split("-")]

    if len(parts) not in (1, 2) or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"run range must look like 5636-7056 or 5636, got {text!r}"
        )

    start = int(parts[0])
    end = int(parts[-1])

    if start > end:
        raise ValueError(
            f"run range starts at {start}, after its end at {end}"
        )

    return start, end


def format_run_range(run_range: RunRange) -> str:
    start, end = run_range
    return f"{start}-{end}"


# ----------------------------------------------------------------------
# NPE per cm from the good-run file
# ----------------------------------------------------------------------

def _plane_station_map() -> dict[str, int]:
    """Plane name (Y1 ... Z4) -> station ID, from STATION_MAP."""
    return {
        plane.upper(): station
        for station, plane in STATION_MAP.items()
    }


def _runs_in_range(
    block: Mapping[str, Any],
    detector: str,
    run_range: RunRange | None,
) -> dict[int, Mapping[str, Any]]:
    """The runs to average: those inside run_range, or, without a range,
    those of the detector's FALLBACK_SETTING. Only good runs are in the
    file, so gaps in the run numbers are normal."""
    runs = {
        int(key): entry
        for key, entry in block.items()
        if key.isdigit()
    }

    if run_range is None:
        setting = FALLBACK_SETTING[detector]
        selected = {
            run: entry
            for run, entry in runs.items()
            if entry.get("set") == setting
        }
        selection = f"fallback setting {setting}"
    else:
        start, end = run_range
        selected = {
            run: entry
            for run, entry in runs.items()
            if start <= run <= end
        }
        selection = f"run range {format_run_range(run_range)}"

    if not selected:
        raise ValueError(
            f"{selection} is not in the good-run list of {detector}"
        )

    return selected




# ----------------------------------------------------------------------
# resolver
# ----------------------------------------------------------------------
def resolve_conversion_factors(
    path: Path,
    detector: str,
    run_range: RunRange | None,
) -> tuple[dict[int, float], dict[str, Any], tuple[str, ...]]:
    """NPE/cm per station for one detector, from the good-run file"""
    try:
        with open(path) as json_file:
            data = json.load(json_file)
            
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the file {path}: {error}") from None

    if detector not in data:
        raise ValueError(f"{detector} is not in the file {path}")

    runs = _runs_in_range(data[detector], detector, run_range)

    if run_range is None:
        range_text = FALLBACK_SETTING[detector]
    else:
        range_text = format_run_range(run_range)

    low, high = NPECM_VALID
    npe_per_cm = {}
    used, null, rejected = {}, {}, {}

    for plane in PLANES:
        values = [entry.get(plane) for entry in runs.values()]
        valid = [v for v in values if v is not None and low <= v <= high]

        used[plane] = len(valid)
        null[plane] = sum(v is None for v in values)
        rejected[plane] = len(values) - len(valid) - null[plane]

        if not valid:
            raise ValueError(
                f"{detector} {plane} has no valid NPE/cm in {range_text} "
                f"({null[plane]} null, {rejected[plane]} outside "
                f"[{low}, {high}])"
            )

        npe_per_cm[plane] = sum(valid) / len(valid)

    settings = dict(sorted(Counter(
        entry.get("set") for entry in runs.values()
    ).items()))

    warnings = []

    if run_range is None:
        warnings.append(
            f"no --run-range: using the {detector} fallback setting "
            f"{range_text} ({len(runs)} runs)"
        )

    if len(settings) > 1:
        listed = ", ".join(f"{name} ({n} runs)" for name, n in settings.items())
        warnings.append(
            f"run range {range_text} spans {len(settings)} settings: {listed}"
        )

    station_of = _plane_station_map()
    factors = {station_of[plane]: value for plane, value in npe_per_cm.items()}

    summary = {
        "run_range": range_text,
        "good_runs": len(runs),
        "npe_per_cm_by_plane": npe_per_cm,
        "runs_used_by_plane": used,
        "runs_null_by_plane": null,
        "runs_rejected_by_plane": rejected,
        "settings_in_range": settings,
        "multi_setting": len(settings) > 1,
    }
    return factors, summary, tuple(warnings)
            
  
# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_conversion_factors(factors: Any) -> dict[int, float]:
    """Check a station -> factor mapping and normalise its types."""
    if not isinstance(factors, Mapping) or not factors:
        raise ValueError(
            "conversion factors must be a non-empty mapping of "
            "station ID to nPE per MeV"
        )

    validated: dict[int, float] = {}

    for station, factor in factors.items():
        if isinstance(station, bool):
            raise ValueError(f"station IDs must be integers, got {station!r}")

        try:
            station_id = operator.index(station)
        except TypeError:
            raise ValueError(
                f"station IDs must be integers, got {station!r}"
            ) from None

        if station_id < 0:
            raise ValueError(
                f"station IDs must be non-negative, got {station_id}"
            )

        try:
            value = float(factor)
        except (TypeError, ValueError):
            raise ValueError(
                f"conversion factor for station {station_id} is not a "
                f"number: {factor!r}"
            ) from None

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"conversion factor for station {station_id} must be "
                f"positive and finite, got {factor!r}"
            )

        validated[station_id] = value

    return validated


def _validate_single_factor(value: float) -> float:
    factor = float(value)

    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(
            f"--pe-per-mev must be positive and finite, got {value!r}"
        )

    return factor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


# ----------------------------------------------------------------------
# From command-line options to a digitizer configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class DigitizationSetup:
    """What the digitizer gets, and what provenance records about it."""

    #: One factor for every plane, or a station -> factor mapping.
    factors: float | dict[int, float]

    #: Recorded in each output file. The same keys appear in both
    #: modes, with None for those that do not apply, so readers never
    #: have to guess which mode produced a file.
    provenance: dict[str, Any]

    #: Inserted into the default output prefix: empty for a single
    #: factor, "_ROSSO_r10061-11918" when factors come from a file.
    tag: str = ""

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def per_station(self) -> bool:
        return isinstance(self.factors, dict)

    
def digitization_setup(
    *,
    pe_per_mev: float | None,
    pe_per_cm_file: str | Path | None,
    detector: str | None,
    run_range: RunRange | None,
    default_pe_per_mev: float,
) -> DigitizationSetup:
    """Apply the rules between the conversion-factor options.

    - Without a file: --pe-per-mev if given, else the default.
      --detector and --run-range are rejected, since they would have no
      effect and could suggest a detector was simulated when it was not.
    - With a file: --detector and --run-range are both required, and the
      file wins over --pe-per-mev, with a warning if both were given.
    """
    if pe_per_cm_file is None:
        stray = [
            name
            for name, value in (
                ("--detector", detector),
                ("--run-range", run_range),
            )
            if value is not None
        ]

        if stray:
            raise ValueError(
                f"{' and '.join(stray)} can only be used together with "
                "--pe-per-cm-file; without a file it would have no "
                "effect"
            )
 
        factor = _validate_single_factor(
            default_pe_per_mev if pe_per_mev is None else pe_per_mev
        )

        return DigitizationSetup(
            factors=factor,
            provenance={
                "digitization_method": "energy",
                "pe_per_mev": factor,
                "pe_per_cm_file": None,
                "pe_per_cm_file_sha256": None,
                "detector": None,
                "run_range": None,
                "pe_per_cm_by_station": None,
                "calibration": None,
            },
        )

    if pe_per_mev is not None:
        raise ValueError(
            "--pe-per-mev and --pe-per-cm-file select two different digitization methods; give only one"
        )

    if detector is None:
        raise ValueError("--pe-per-cm-file requires --detector")


    if detector not in DETECTORS:
        raise ValueError(
            f"unknown detector {detector!r}; expected one of "
            f"{', '.join(DETECTORS)}"
        )

    path = Path(pe_per_cm_file).expanduser().resolve()

    if not path.is_file():
        raise ValueError(f"conversion-factor file not found: {path}")

 

    factors, summary, warnings = resolve_conversion_factors(
        path, detector, run_range
    )
    factors = validate_conversion_factors(factors)


    return DigitizationSetup(
        factors=factors,
        provenance={
            "digitization_method": "npe_per_cm",
            "pe_per_mev": None,
            "pe_per_cm_file": str(path),
            "pe_per_cm_file_sha256": file_sha256(path),
            "detector": detector,
            "run_range": summary["run_range"],
            "pe_per_cm_by_station": {
                str(station): factor
                for station, factor in sorted(factors.items())
            },
            "calibration": summary,
        },
        tag=f"_{detector}_r{summary['run_range']}",
        warnings=warnings,
    )
