from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterRecord:

    process_id: int
    event: int

    station: int
    cluster_id: int

    cluster_multiplicity: int

    cluster_energy: float
    cluster_position: float
    cluster_size: int

    cluster_time: float
    cluster_time_spread: float

    strip_ids: list[int]
    strip_energies: list[float]
