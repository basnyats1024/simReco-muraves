# models/tracking_event.py

from __future__ import annotations

from dataclasses import dataclass, field

from .cluster_record import ClusterRecord


@dataclass
class TrackingEvent:
    process_id: int
    event: int

    y1: list[ClusterRecord] = field(default_factory=list)
    y2: list[ClusterRecord] = field(default_factory=list)
    y3: list[ClusterRecord] = field(default_factory=list)
    y4: list[ClusterRecord] = field(default_factory=list)

    z1: list[ClusterRecord] = field(default_factory=list)
    z2: list[ClusterRecord] = field(default_factory=list)
    z3: list[ClusterRecord] = field(default_factory=list)
    z4: list[ClusterRecord] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return (
            len(self.y1) + len(self.y2) + len(self.y3) + len(self.y4)
            + len(self.z1) + len(self.z2) + len(self.z3) + len(self.z4)
        )

