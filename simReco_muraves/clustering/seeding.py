# simReco_muraves/clustering/seeding.py

"""Per-plane random seeds for clusterization.

Two random processes run during clusterization: Poisson digitization
of each bar, and the experiment's random smearing of single-strip
cluster positions. Both are seeded per (ProcessID, Event, Station)
from one master seed, rather than drawn from long streams consumed in
processing order.

That buys three things.

Reproducibility. The same input, master seed and code give identical
output. Without an explicit generator, the experiment's ClusterLists
falls back to random.SystemRandom, which cannot be seeded at all.

Order independence. Results do not depend on the order in which
events are processed, so splitting a production by ProcessID or
moving chunk boundaries cannot change them.

Isolation between planes. Changing one plane's inputs, such as its
conversion factor, changes only that plane's random numbers. With a
single stream, everything after the first changed draw would be
re-randomised, because numpy's Poisson sampler consumes a variable
number of uniforms depending on its mean.

Seeds are derived with numpy's SeedSequence, which mixes its inputs
thoroughly. That matters for smearing: the xorshift generator used
there usually contributes only its first output per plane, and that
output is a linear function of the seed, so naive seeds such as
master + station would give correlated smearing across planes.

Reproducibility holds within one software environment. NumPy does not
guarantee identical sampling streams across versions, which is one
reason provenance records the numpy version.
"""

from __future__ import annotations

import numpy as np


#: Recorded in provenance. Change it whenever the derivation below
#: changes, since outputs produced under different schemes are not
#: comparable draw for draw.
SEEDING_SCHEME = "per-plane-v1"

_STREAM_DIGITIZER = 0
_STREAM_SMEARING = 1


def _entropy(
    master_seed: int,
    stream: int,
    process_id: int,
    event: int,
    station: int,
) -> list[int]:
    values = [
        int(master_seed),
        stream,
        int(process_id),
        int(event),
        int(station),
    ]

    if any(value < 0 for value in values):
        raise ValueError(
            "seed inputs must be non-negative: "
            f"master_seed={master_seed}, process_id={process_id}, "
            f"event={event}, station={station}"
        )

    return values


def digitizer_generator(
    master_seed: int,
    process_id: int,
    event: int,
    station: int,
) -> np.random.Generator:
    """Generator for the Poisson digitization of one plane in one event."""
    return np.random.default_rng(
        np.random.SeedSequence(
            _entropy(
                master_seed,
                _STREAM_DIGITIZER,
                process_id,
                event,
                station,
            )
        )
    )


def smearing_seed(
    master_seed: int,
    process_id: int,
    event: int,
    station: int,
) -> int:
    """32-bit seed for the single-strip smearing of one plane in one event."""
    state = np.random.SeedSequence(
        _entropy(
            master_seed,
            _STREAM_SMEARING,
            process_id,
            event,
            station,
        )
    ).generate_state(1, dtype=np.uint32)

    return int(state[0])
