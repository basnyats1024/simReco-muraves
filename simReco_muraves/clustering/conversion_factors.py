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


DETECTORS = ("NERO", "ROSSO", "BLU")

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
# The stub
# ----------------------------------------------------------------------

def resolve_conversion_factors(
    path: Path,
    detector: str,
    run_range: RunRange,
) -> dict[int, float]:
    """Per-station conversion factors for one detector and run range.

    NOT IMPLEMENTED YET. This is where the factors file is read.

    Contract
    --------
    Return a mapping from **station ID** to a conversion factor in
    **nPE per MeV**. Station IDs are the integers the digitizer sees on
    every BarHit (0, 1, 2, 3, 8, 9, 10, 11 in the current geometry).
    The station-to-plane mapping is STATION_MAP in
    simReco_muraves/tracking/event_builder_adapter.py: stations 0-3 are
    Y1-Y4 and 8-11 are Z1-Z4. Use it rather than keeping a second copy.

    - The file format is yours to design. If it is keyed by plane name
      (Z1 ... Y4) or anything else, translating to station IDs happens
      here, so the digitizer never needs to know about names.
    - Return a factor for every station that will appear in the input.
      A missing station is not defaulted: digitization stops with an
      error naming it.
    - How a run range that spans more than one calibration is handled
      is yours to decide. Raising is the safe default; silently picking
      one or averaging is not.
    - The result is passed through validate_conversion_factors(), so
      malformed values are caught even if this function lets them by.

    The caller has already checked that ``path`` exists, that
    ``detector`` is one of DETECTORS, and that ``run_range`` is a valid
    (start, end) pair with start <= end.
    """
    raise NotImplementedError(
        "reading per-station conversion factors from a file is not "
        "implemented yet. Implement resolve_conversion_factors() in "
        "simReco_muraves/clustering/conversion_factors.py; its docstring "
        "describes what it must return. Until then, use --pe-per-mev."
    )


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
    pe_per_mev_file: str | Path | None,
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
    if pe_per_mev_file is None:
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
                "--pe-per-mev-file; without a file it would have no "
                "effect"
            )

        factor = _validate_single_factor(
            default_pe_per_mev if pe_per_mev is None else pe_per_mev
        )

        return DigitizationSetup(
            factors=factor,
            provenance={
                "pe_per_mev": factor,
                "pe_per_mev_file": None,
                "pe_per_mev_file_sha256": None,
                "detector": None,
                "run_range": None,
                "pe_per_mev_by_station": None,
            },
        )

    missing = [
        name
        for name, value in (
            ("--detector", detector),
            ("--run-range", run_range),
        )
        if value is None
    ]

    if missing:
        raise ValueError(
            f"--pe-per-mev-file requires {' and '.join(missing)}"
        )

    if detector not in DETECTORS:
        raise ValueError(
            f"unknown detector {detector!r}; expected one of "
            f"{', '.join(DETECTORS)}"
        )

    path = Path(pe_per_mev_file).expanduser().resolve()

    if not path.is_file():
        raise ValueError(f"conversion-factor file not found: {path}")

    warnings: tuple[str, ...] = ()

    if pe_per_mev is not None:
        warnings = (
            f"--pe-per-mev {pe_per_mev} is ignored: "
            "--pe-per-mev-file takes precedence",
        )

    factors = validate_conversion_factors(
        resolve_conversion_factors(path, detector, run_range)
    )

    range_text = format_run_range(run_range)

    return DigitizationSetup(
        factors=factors,
        provenance={
            "pe_per_mev": None,
            "pe_per_mev_file": str(path),
            "pe_per_mev_file_sha256": file_sha256(path),
            "detector": detector,
            "run_range": range_text,
            "pe_per_mev_by_station": {
                str(station): factor
                for station, factor in sorted(factors.items())
            },
        },
        tag=f"_{detector}_r{range_text}",
        warnings=warnings,
    )
