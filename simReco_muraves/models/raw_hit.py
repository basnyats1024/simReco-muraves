from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RawHit:
    process_id: int
    event: int

    station: int
    module: int
    bar: int

    edep_mev: float
    time_ns: float

    track_id: int
    parent_id: int
    pdg: int
