# simReco_muraves/clustering/tests/test_conversion_factors.py

"""Conversion-factor options and the per-station digitizer.

Covers the rules between --pe-per-mev, --pe-per-mev-file, --detector
and --run-range; validation of resolved factors; the digitizer's
per-station path; and the property per-plane factor studies rely on:
changing one station's factor leaves every other station's random
numbers untouched.
"""

from __future__ import annotations

import pytest

from simReco_muraves.clustering import conversion_factors as cf
from simReco_muraves.clustering.sim_clusterizer import (
    BarHit,
    PoissonDigitizer,
)


# ----------------------------------------------------------------------
# Run ranges
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("5636-7056", (5636, 7056)),
        ("5636", (5636, 5636)),
        (" 5636 - 7056 ", (5636, 7056)),
        ("7056-7056", (7056, 7056)),
    ],
)
def test_run_range_is_parsed(text, expected):
    assert cf.parse_run_range(text) == expected


@pytest.mark.parametrize(
    "text",
    ["7056-5636", "abc", "-5", "1-2-3", "", "56a-60", "5636-"],
)
def test_malformed_run_range_is_rejected(text):
    with pytest.raises(ValueError):
        cf.parse_run_range(text)


def test_run_range_round_trips():
    assert cf.format_run_range(cf.parse_run_range("10061-11918")) == "10061-11918"


# ----------------------------------------------------------------------
# Validation of resolved factors
# ----------------------------------------------------------------------

def test_valid_factors_are_normalised():
    assert cf.validate_conversion_factors({0: 8, 8: 9.5}) == {0: 8.0, 8: 9.5}


@pytest.mark.parametrize(
    "factors",
    [
        {},
        {-1: 8.0},
        {"Z1": 8.0},
        {True: 8.0},
        {0: 0.0},
        {0: -1.0},
        {0: float("nan")},
        {0: float("inf")},
        {0: "not a number"},
        [8.0, 8.0],
    ],
)
def test_invalid_factors_are_rejected(factors):
    with pytest.raises(ValueError):
        cf.validate_conversion_factors(factors)


def test_resolver_is_still_a_stub(tmp_path):
    with pytest.raises(NotImplementedError, match="resolve_conversion_factors"):
        cf.resolve_conversion_factors(tmp_path / "f.json", "ROSSO", (1, 2))


# ----------------------------------------------------------------------
# Rules between the options
# ----------------------------------------------------------------------

RESOLVED = {0: 8.0, 1: 9.0, 8: 7.5}


def setup(**overrides):
    values = dict(
        pe_per_mev=None,
        pe_per_mev_file=None,
        detector=None,
        run_range=None,
        default_pe_per_mev=8.5,
    )
    values.update(overrides)
    return cf.digitization_setup(**values)


@pytest.fixture
def factors_file(tmp_path, monkeypatch):
    """A file to point at, with the stub replaced by a fixed answer."""
    path = tmp_path / "factors.json"
    path.write_text('{"placeholder": true}')
    monkeypatch.setattr(
        cf,
        "resolve_conversion_factors",
        lambda file, detector, run_range: dict(RESOLVED),
    )
    return path


def with_file(factors_file, **overrides):
    values = dict(
        pe_per_mev_file=factors_file,
        detector="ROSSO",
        run_range=(10061, 11918),
    )
    values.update(overrides)
    return setup(**values)


def test_default_factor_when_nothing_given():
    result = setup()

    assert result.factors == 8.5
    assert not result.per_station
    assert result.tag == ""
    assert result.warnings == ()


def test_explicit_single_factor():
    assert setup(pe_per_mev=4.0).factors == 4.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_bad_single_factor_is_rejected(value):
    with pytest.raises(ValueError):
        setup(pe_per_mev=value)


@pytest.mark.parametrize(
    "overrides",
    [{"detector": "ROSSO"}, {"run_range": (1, 2)}],
)
def test_detector_or_run_range_without_file_is_rejected(overrides):
    with pytest.raises(ValueError, match="can only be used together"):
        setup(**overrides)


@pytest.mark.parametrize(
    "missing, message",
    [
        ({"detector": None}, "--detector"),
        ({"run_range": None}, "--run-range"),
        ({"detector": None, "run_range": None}, "--detector and --run-range"),
    ],
)
def test_file_requires_detector_and_run_range(factors_file, missing, message):
    with pytest.raises(ValueError, match=message):
        with_file(factors_file, **missing)


def test_unknown_detector_is_rejected(factors_file):
    with pytest.raises(ValueError, match="unknown detector"):
        with_file(factors_file, detector="GIALLO")


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        setup(
            pe_per_mev_file=tmp_path / "absent.json",
            detector="ROSSO",
            run_range=(1, 2),
        )


def test_file_mode(factors_file):
    result = with_file(factors_file)

    assert result.per_station
    assert result.factors == RESOLVED
    assert result.tag == "_ROSSO_r10061-11918"
    assert result.warnings == ()
    assert result.provenance["detector"] == "ROSSO"
    assert result.provenance["run_range"] == "10061-11918"
    assert result.provenance["pe_per_mev"] is None
    assert result.provenance["pe_per_mev_by_station"] == {
        "0": 8.0,
        "1": 9.0,
        "8": 7.5,
    }
    assert len(result.provenance["pe_per_mev_file_sha256"]) == 64


def test_file_wins_over_explicit_factor(factors_file):
    result = with_file(factors_file, pe_per_mev=4.0)

    assert result.per_station
    assert result.factors == RESOLVED
    assert len(result.warnings) == 1
    assert "ignored" in result.warnings[0]


def test_resolved_factors_are_validated(factors_file, monkeypatch):
    monkeypatch.setattr(
        cf,
        "resolve_conversion_factors",
        lambda file, detector, run_range: {0: -1.0},
    )

    with pytest.raises(ValueError, match="positive and finite"):
        with_file(factors_file)


def test_stub_stops_file_mode_until_implemented(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text("{}")

    with pytest.raises(NotImplementedError):
        setup(pe_per_mev_file=path, detector="ROSSO", run_range=(1, 2))


def test_provenance_has_the_same_keys_in_both_modes(factors_file):
    assert set(setup().provenance) == set(with_file(factors_file).provenance)


def test_checksum_follows_file_content(factors_file):
    before = with_file(factors_file).provenance["pe_per_mev_file_sha256"]
    factors_file.write_text('{"edited": true}')
    after = with_file(factors_file).provenance["pe_per_mev_file_sha256"]

    assert before != after


# ----------------------------------------------------------------------
# The per-station digitizer
# ----------------------------------------------------------------------

def hit(station, strip=5, energy_mev=2.0):
    return BarHit(
        process_id=1,
        event=100,
        station=station,
        module=strip // 32,
        bar=strip % 32,
        strip=strip,
        energy_mev=energy_mev,
        t_min_ns=1.0,
        t_max_ns=1.0,
        multiplicity=1,
    )


def event_hits():
    return [
        hit(station, strip, 2.0 + 0.1 * strip)
        for station in (0, 1, 8)
        for strip in (5, 6, 40)
    ]


def by_bar(digitized):
    return {(d.station, d.strip): d.observed_npe for d in digitized}


def test_mapping_applies_each_station_its_own_factor():
    digitized = PoissonDigitizer(
        pe_per_mev={0: 8.0, 8: 4.0},
        rng_seed=42,
    ).digitize([hit(0), hit(8)])

    assert digitized[0].expected_npe == pytest.approx(16.0)
    assert digitized[1].expected_npe == pytest.approx(8.0)


def test_single_factor_path_is_unchanged():
    digitized = PoissonDigitizer(pe_per_mev=8.5).digitize([hit(3)])

    assert digitized[0].expected_npe == pytest.approx(17.0)


def test_missing_station_is_an_error_naming_it():
    digitizer = PoissonDigitizer(pe_per_mev={0: 8.0}, rng_seed=42)

    with pytest.raises(ValueError, match="station 8"):
        digitizer.digitize([hit(0), hit(8)])


def test_uniform_mapping_matches_single_factor_exactly():
    """Switching to a file must not itself change the random numbers."""
    hits = event_hits()

    single = PoissonDigitizer(pe_per_mev=8.5, rng_seed=42).digitize(hits)
    mapping = PoissonDigitizer(
        pe_per_mev={0: 8.5, 1: 8.5, 8: 8.5},
        rng_seed=42,
    ).digitize(hits)

    assert by_bar(single) == by_bar(mapping)


def test_changing_one_station_factor_leaves_others_untouched():
    hits = event_hits()
    base = {0: 8.0, 1: 8.0, 8: 8.0}
    changed = dict(base)
    changed[0] = 4.0

    before = by_bar(PoissonDigitizer(pe_per_mev=base, rng_seed=42).digitize(hits))
    after = by_bar(PoissonDigitizer(pe_per_mev=changed, rng_seed=42).digitize(hits))

    for (station, strip), observed in before.items():
        if station != 0:
            assert after[(station, strip)] == observed

    assert any(
        after[key] != before[key]
        for key in before
        if key[0] == 0
    )


# ----------------------------------------------------------------------
# Command-line parsing
# ----------------------------------------------------------------------

def cli():
    pytest.importorskip("uproot")
    from simReco_muraves.cli import run_clusterization

    return run_clusterization


def test_cli_pe_per_mev_defaults_to_none():
    assert cli().parse_arguments(["hits.root"]).pe_per_mev is None


def test_cli_detector_is_case_insensitive():
    arguments = cli().parse_arguments(
        [
            "hits.root",
            "--pe-per-mev-file", "factors.json",
            "--detector", "rosso",
            "--run-range", "10061-11918",
        ]
    )

    assert arguments.detector == "ROSSO"
    assert arguments.run_range == (10061, 11918)


@pytest.mark.parametrize(
    "extra",
    [
        ["--run-range", "11918-10061"],
        ["--run-range", "abc"],
        ["--detector", "GIALLO"],
    ],
)
def test_cli_rejects_bad_values(extra):
    with pytest.raises(SystemExit):
        cli().parse_arguments(["hits.root", *extra])
