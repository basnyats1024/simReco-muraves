# simReco_muraves/clustering/tests/test_seeding.py

"""Reproducibility of clusterization.

These encode the properties the per-plane seeding exists to provide:
same seed gives identical output, a different seed gives different
output, results do not depend on processing order, and changing one
plane's inputs leaves every other plane's random numbers untouched.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from simReco_muraves.clustering.seeding import (
    SEEDING_SCHEME,
    digitizer_generator,
    smearing_seed,
)
from simReco_muraves.clustering.sim_clusterizer import (
    BarHit,
    Plane64,
    PoissonDigitizer,
    SimClusterizer,
)


# ----------------------------------------------------------------------
# Seed derivation
# ----------------------------------------------------------------------

def test_scheme_is_named():
    assert SEEDING_SCHEME


def test_smearing_seed_is_deterministic():
    assert smearing_seed(42, 1, 100, 3) == smearing_seed(42, 1, 100, 3)


def test_smearing_seed_depends_on_every_key():
    base = smearing_seed(42, 1, 100, 3)

    assert base != smearing_seed(43, 1, 100, 3)
    assert base != smearing_seed(42, 2, 100, 3)
    assert base != smearing_seed(42, 1, 101, 3)
    assert base != smearing_seed(42, 1, 100, 8)


def test_smearing_seed_fits_32_bits():
    assert 0 <= smearing_seed(42, 1, 100, 3) < 2**32


def test_digitizer_generator_is_deterministic():
    first = digitizer_generator(42, 1, 100, 3).poisson(20.0, size=50)
    second = digitizer_generator(42, 1, 100, 3).poisson(20.0, size=50)

    assert first.tolist() == second.tolist()


@pytest.mark.parametrize(
    "arguments",
    [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)],
)
def test_negative_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        smearing_seed(*arguments)


# ----------------------------------------------------------------------
# Digitizer
# ----------------------------------------------------------------------

def make_hit(process_id, event, station, strip, energy_mev):
    return BarHit(
        process_id=process_id,
        event=event,
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
        make_hit(1, 100, station, strip, 2.0 + 0.1 * strip)
        for station in (0, 1, 8)
        for strip in (5, 6, 40)
    ]


def by_bar(digitized):
    return {(hit.station, hit.strip): hit.observed_npe for hit in digitized}


def test_seeded_digitizer_is_reproducible():
    first = PoissonDigitizer(rng_seed=42).digitize(event_hits())
    second = PoissonDigitizer(rng_seed=42).digitize(event_hits())

    assert by_bar(first) == by_bar(second)


def test_seed_changes_the_draws():
    first = PoissonDigitizer(rng_seed=42).digitize(event_hits())
    second = PoissonDigitizer(rng_seed=43).digitize(event_hits())

    assert by_bar(first) != by_bar(second)


def test_result_does_not_depend_on_hit_order():
    hits = event_hits()

    forward = PoissonDigitizer(rng_seed=42).digitize(hits)
    backward = PoissonDigitizer(rng_seed=42).digitize(list(reversed(hits)))

    assert by_bar(forward) == by_bar(backward)


def test_output_preserves_input_order():
    hits = event_hits()
    digitized = PoissonDigitizer(rng_seed=42).digitize(hits)

    assert len(digitized) == len(hits)
    assert all(
        result.bar_hit is hit
        for result, hit in zip(digitized, hits)
    )


def test_changing_one_plane_leaves_other_planes_untouched():
    """The property that keeps per-plane factor studies clean.

    Halving the energy on station 0 stands in for changing that
    plane's conversion factor. Every bar on the other stations must
    draw exactly the same number as before.
    """
    hits = event_hits()
    changed = [
        replace(hit, energy_mev=hit.energy_mev * 0.5)
        if hit.station == 0
        else hit
        for hit in hits
    ]

    before = by_bar(PoissonDigitizer(rng_seed=42).digitize(hits))
    after = by_bar(PoissonDigitizer(rng_seed=42).digitize(changed))

    for (station, strip), observed in before.items():
        if station != 0:
            assert after[(station, strip)] == observed


def test_unseeded_digitizer_still_works():
    digitized = PoissonDigitizer(rng_seed=None).digitize(event_hits())

    assert len(digitized) == 9
    assert all(hit.observed_npe >= 0 for hit in digitized)


# ----------------------------------------------------------------------
# Single-strip smearing
# ----------------------------------------------------------------------

EVENTS = range(30)


def single_strip_plane(event):
    """A plane with one isolated strip, enough for a single-strip cluster."""
    plane = Plane64(process_id=1, event=event, station=0)
    plane.deposits[20] = 40.0
    return plane


def positions(clusterizer):
    return [
        list(clusterizer.cluster_plane(single_strip_plane(event)).ClustersPositions)
        for event in EVENTS
    ]


@pytest.fixture
def smearing_is_exercised():
    collection = SimClusterizer(rng_seed=42).cluster_plane(single_strip_plane(0))

    if 1 not in list(collection.ClustersSize):
        pytest.skip(
            "test plane produced no single-strip cluster, so smearing "
            "is not exercised; adjust single_strip_plane()"
        )


def test_seeded_smearing_is_reproducible(smearing_is_exercised):
    assert positions(SimClusterizer(rng_seed=42)) == positions(
        SimClusterizer(rng_seed=42)
    )


def test_smearing_seed_changes_positions(smearing_is_exercised):
    assert positions(SimClusterizer(rng_seed=42)) != positions(
        SimClusterizer(rng_seed=43)
    )


def test_smearing_varies_between_events(smearing_is_exercised):
    distinct = {tuple(position) for position in positions(SimClusterizer(rng_seed=42))}

    assert len(distinct) > 1
