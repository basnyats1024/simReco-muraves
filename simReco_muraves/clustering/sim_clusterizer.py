from __future__ import annotations

from collections import defaultdict
import numpy as np
from simReco_muraves.models.cluster_record import ClusterRecord

# Adjust this import to wherever CreateClusterList actually lives
from simReco_muraves.clustering.ClusterLists import CreateClusterList

from dataclasses import dataclass, field

@dataclass
class BarHit:
    process_id: int
    event: int
    station: int
    module: int
    bar: int
    strip: int
    energy_mev: float
    t_min_ns: float
    t_max_ns: float
    multiplicity: int

    @property
    def dt_ns(self) -> float:
        return self.t_max_ns - self.t_min_ns

@dataclass
class DigitizedBarHit:
    bar_hit: BarHit
    expected_npe: float
    observed_npe: float

    @property
    def process_id(self) -> int:
        return self.bar_hit.process_id

    @property
    def strip(self) -> int:
        return self.bar_hit.strip

    @property
    def station(self) -> int:
        return self.bar_hit.station

    @property
    def event(self) -> int:
        return self.bar_hit.event

    @property
    def t_min_ns(self) -> float:
        return self.bar_hit.t_min_ns


@dataclass
class Plane64:
    process_id: int
    event: int
    station: int
    deposits: list[float] = field(default_factory=lambda: [0.0] * 64)
    times: list[float] = field(default_factory=lambda: [0.0] * 64)
    occupied_strips: int = 0



class SimClusterizer:
    """
    Thin wrapper around the official MURAVES clusterizer.
    """

    def __init__(
        self,
        cluster_threshold: float = 6.0,
        single_strip_threshold: float = 10.0,
        adjacent_threshold: float = 2.0,
    ):

        self.cluster_threshold = cluster_threshold
        self.single_strip_threshold = single_strip_threshold
        self.adjacent_threshold = adjacent_threshold

    def cluster_plane(
        self,
        plane: Plane64,
    ):

        return CreateClusterList(
            Deposits=plane.deposits,

            EnergyThreshold_clusterStrip=
                self.cluster_threshold,

            EnergyThreshold_singleStrip=
                self.single_strip_threshold,

            AdStripsThEnergy_singleStripCl=
                self.adjacent_threshold,

            Texp1=1.0,
            Texp2=1.0,

            TriggerMask1=list(range(32)),
            TriggerMask2=list(range(32)),
        )
def build_planes(digitized_hits):
    """
    Build one Plane64 object per (event, station).
    """

    grouped = defaultdict(list)

    for hit in digitized_hits:

        grouped[
            (
                hit.process_id,
                hit.event,
                hit.station,
            )
        ].append(hit)

    planes = []

    for (process_id, event, station), hits in grouped.items():

        plane = Plane64(
            process_id=process_id,
            event=event,
            station=station,
        )

        occupied = 0

        for hit in hits:

            strip = hit.strip

            if strip < 0 or strip >= 64:
                raise ValueError(
                    f"Invalid strip number: {strip}"
                )

            plane.deposits[strip] += hit.observed_npe

            plane.times[strip] = hit.t_min_ns

            occupied += 1

        plane.occupied_strips = occupied

        planes.append(plane)

    return planes
def module_bar_to_strip(module, bar):
    """
    Convert (module, bar) into global strip number.

    module 0 -> strips 0..31
    module 1 -> strips 32..63
    """

    if module == 0:
        return bar

    if module == 1:
        return bar + 32

    raise ValueError(
        f"Unexpected module number: {module}"
    )


def aggregate_bar_hits(raw_hits):
    """
    Aggregate Geant4 HitData entries into bar-level responses.

    Grouping key:
        (event, station, module, bar)

    Energy:
        E_bar = sum(Edep)

    Timing:
        t_min = min(hit times)
        t_max = max(hit times)

    Multiplicity:
        number of contributing hits
    """

    grouped = defaultdict(list)

    for hit in raw_hits:

        key = (
            hit.process_id,
            hit.event,
            hit.station,
            hit.module,
            hit.bar,
        )

        grouped[key].append(hit)

    aggregated = []

    for key, hits in grouped.items():

        process_id, event, station, module, bar = key

        strip = module_bar_to_strip(
            module,
            bar,
        )

        energy_mev = sum(
            h.edep_mev for h in hits
        )

        t_min_ns = min(
            h.time_ns for h in hits
        )

        t_max_ns = max(
            h.time_ns for h in hits
        )

        aggregated.append(
            BarHit(
                process_id=process_id,
                event=event,
                station=station,
                module=module,
                bar=bar,
                strip=strip,
                energy_mev=energy_mev,
                t_min_ns=t_min_ns,
                t_max_ns=t_max_ns,
                multiplicity=len(hits),
            )
        )

    return aggregated
class PoissonDigitizer:
    """
    First-pass digitizer.

    expected_nPE = pe_per_mev * E(MeV)

    observed_nPE ~ Poisson(expected_nPE)
    """

    def __init__(
        self,
        pe_per_mev=8.5,
        rng_seed=None,
    ):

        self.pe_per_mev = pe_per_mev
        self.rng = np.random.default_rng(rng_seed)

    def digitize_bar(self, bar_hit):

        expected_npe = (
            self.pe_per_mev
            * bar_hit.energy_mev
        )

        observed_npe = float(
            self.rng.poisson(expected_npe)
        )

        return DigitizedBarHit(
            bar_hit=bar_hit,
            expected_npe=expected_npe,
            observed_npe=observed_npe,
        )

    def digitize(self, bar_hits):

        return [
            self.digitize_bar(hit)
            for hit in bar_hits
        ]
def cluster_collection_to_records(
    event,
    station,
    cluster_collection,
    plane,
):

    records = []

    n_clusters = len(
        cluster_collection.ClustersEnergy
    )

    cluster_multiplicity = n_clusters

    for idx in range(n_clusters):

        strip_ids = [
            int(x)
            for x in cluster_collection.StripsID[idx]
        ]

        strip_energies = [
            float(x)
            for x in cluster_collection.StripsEnergy[idx]
        ]

        strip_times = [
            plane.times[strip]
            for strip in strip_ids
        ]

        cluster_time = min(strip_times)

        cluster_time_spread = (
            max(strip_times)
            - min(strip_times)
        )

        records.append(

            ClusterRecord(

                process_id=plane.process_id,

                event=event,
                station=station,
                cluster_id=idx,

                cluster_multiplicity=
                    cluster_multiplicity,

                cluster_energy=float(
                    cluster_collection.ClustersEnergy[idx]
                ),

                cluster_position=float(
                    cluster_collection.ClustersPositions[idx]
                ),

                cluster_size=int(
                    cluster_collection.ClustersSize[idx]
                ),

                cluster_time=float(
                    cluster_time
                ),

                cluster_time_spread=float(
                    cluster_time_spread
                ),

                strip_ids=strip_ids,

                strip_energies=strip_energies,
            )
        )

    return records
