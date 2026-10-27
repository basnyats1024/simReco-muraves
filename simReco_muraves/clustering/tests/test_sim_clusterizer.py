# simReco_muraves/clustering/tests/test_sim_clusterizer.py

"""Bar aggregation, plane building, cluster records, clusterizer inputs.

These cover the stages around digitization: how Geant4 hits become bar
responses, how digitized bars become 64-strip planes, how the
experiment's cluster collection becomes ClusterRecords, and what the
simulation hands to the experiment's CreateClusterList.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from simReco_muraves.clustering import sim_clusterizer as sc
from simReco_muraves.clustering.ClusterLists import ClusterCollection


def raw_hit(process_id=1, event=10, station=0, module=0, bar=5, edep=1.0, time=2.0):
    return SimpleNamespace(
        process_id=process_id,
        event=event,
        station=station,
        module=module,
        bar=bar,
        edep_mev=edep,
        time_ns=time,
    )


def digitized(station, strip, npe, process_id=1, event=10, time=2.0):
    bar_hit = sc.BarHit(
        process_id=process_id,
        event=event,
        station=station,
        module=strip // 32,
        bar=strip % 32,
        strip=strip,
        energy_mev=npe / 8.5,
        t_min_ns=time,
        t_max_ns=time,
        multiplicity=1,
    )
    return sc.DigitizedBarHit(bar_hit=bar_hit, expected_npe=npe, observed_npe=npe)


# ----------------------------------------------------------------------
# Strip numbering
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "module, bar, strip",
    [(0, 0, 0), (0, 5, 5), (0, 31, 31), (1, 0, 32), (1, 5, 37), (1, 31, 63)],
)
def test_module_and_bar_map_to_a_global_strip(module, bar, strip):
    assert sc.module_bar_to_strip(module, bar) == strip


def test_unknown_module_is_rejected():
    with pytest.raises(ValueError, match="module"):
        sc.module_bar_to_strip(2, 0)


# ----------------------------------------------------------------------
# Bar aggregation
# ----------------------------------------------------------------------

def test_hits_in_one_bar_are_merged():
    bars = sc.aggregate_bar_hits(
        [raw_hit(edep=1.0, time=3.0), raw_hit(edep=2.5, time=1.5)]
    )

    assert len(bars) == 1
    assert bars[0].energy_mev == pytest.approx(3.5)
    assert bars[0].t_min_ns == 1.5
    assert bars[0].t_max_ns == 3.0
    assert bars[0].dt_ns == pytest.approx(1.5)
    assert bars[0].multiplicity == 2


@pytest.mark.parametrize(
    "difference",
    [{"bar": 6}, {"module": 1}, {"station": 8}, {"event": 11}],
)
def test_hits_in_different_bars_stay_separate(difference):
    bars = sc.aggregate_bar_hits([raw_hit(), raw_hit(**difference)])

    assert len(bars) == 2


def test_same_event_number_in_different_processes_is_not_merged():
    """The Phase I bookkeeping bug, at bar level."""
    bars = sc.aggregate_bar_hits([raw_hit(process_id=1), raw_hit(process_id=2)])

    assert sorted(bar.process_id for bar in bars) == [1, 2]


def test_bar_gets_its_global_strip():
    assert sc.aggregate_bar_hits([raw_hit(module=1, bar=3)])[0].strip == 35


def test_no_hits_gives_no_bars():
    assert sc.aggregate_bar_hits([]) == []


# ----------------------------------------------------------------------
# Plane building
# ----------------------------------------------------------------------

def test_one_plane_per_process_event_and_station():
    planes = sc.build_planes(
        [digitized(0, 3, 12.0), digitized(0, 40, 7.0), digitized(8, 3, 9.0)]
    )

    assert sorted(plane.station for plane in planes) == [0, 8]


def test_plane_holds_64_strips_with_deposits_in_place():
    planes = sc.build_planes(
        [digitized(0, 3, 12.0, time=4.0), digitized(0, 40, 7.0)]
    )
    plane = planes[0]

    assert len(plane.deposits) == 64
    assert plane.deposits[3] == 12.0
    assert plane.deposits[40] == 7.0
    assert sum(plane.deposits) == pytest.approx(19.0)
    assert plane.times[3] == 4.0
    assert plane.occupied_strips == 2


def test_planes_keep_processes_apart():
    planes = sc.build_planes(
        [digitized(0, 3, 12.0, process_id=1), digitized(0, 3, 12.0, process_id=2)]
    )

    assert sorted(plane.process_id for plane in planes) == [1, 2]


def test_strip_outside_the_plane_is_rejected():
    with pytest.raises(ValueError, match="strip"):
        sc.build_planes([digitized(0, 64, 12.0)])


# ----------------------------------------------------------------------
# Cluster records
# ----------------------------------------------------------------------

def collection():
    return ClusterCollection(
        ClustersEnergy=[30.0, 12.0],
        ClustersPositions=[0.1, -0.2],
        StripsEnergy=[[18.0, 12.0], [12.0]],
        StripsID=[[4, 5], [20]],
        ClustersSize=[2, 1],
    )


def plane_with_times():
    plane = sc.Plane64(process_id=3, event=10, station=8)
    plane.times[4] = 5.0
    plane.times[5] = 7.0
    plane.times[20] = 2.0
    return plane


def test_records_carry_the_collection_fields():
    records = sc.cluster_collection_to_records(
        event=10,
        station=8,
        cluster_collection=collection(),
        plane=plane_with_times(),
    )

    assert len(records) == 2

    first = records[0]

    assert (first.process_id, first.event, first.station) == (3, 10, 8)
    assert first.cluster_id == 0
    assert first.cluster_multiplicity == 2
    assert first.cluster_energy == 30.0
    assert first.cluster_position == pytest.approx(0.1)
    assert first.cluster_size == 2
    assert first.strip_ids == [4, 5]
    assert first.strip_energies == [18.0, 12.0]


def test_record_time_is_the_earliest_strip_and_spread_the_range():
    records = sc.cluster_collection_to_records(
        event=10,
        station=8,
        cluster_collection=collection(),
        plane=plane_with_times(),
    )

    assert records[0].cluster_time == 5.0
    assert records[0].cluster_time_spread == 2.0
    assert records[1].cluster_time == 2.0
    assert records[1].cluster_time_spread == 0.0


def test_process_id_comes_from_the_plane():
    """The function takes no process ID; it must come from the plane."""
    records = sc.cluster_collection_to_records(
        event=10,
        station=8,
        cluster_collection=collection(),
        plane=plane_with_times(),
    )

    assert {record.process_id for record in records} == {3}


def test_empty_collection_gives_no_records():
    assert sc.cluster_collection_to_records(
        event=10,
        station=8,
        cluster_collection=ClusterCollection(),
        plane=sc.Plane64(process_id=1, event=10, station=8),
    ) == []


# ----------------------------------------------------------------------
# What the clusterizer hands to the experiment's code
# ----------------------------------------------------------------------

def capture_clusterizer_call(monkeypatch, clusterizer, plane):
    captured = {}

    def fake_create_cluster_list(**arguments):
        captured.update(arguments)
        return ClusterCollection()

    monkeypatch.setattr(sc, "CreateClusterList", fake_create_cluster_list)
    clusterizer.cluster_plane(plane)

    return captured


def test_default_thresholds_match_the_collaboration_config():
    """6.0 / 10.0 / 2.0 nPE, as cluster_thresholds in reco_parameters.json."""
    clusterizer = sc.SimClusterizer()

    assert (
        clusterizer.cluster_threshold,
        clusterizer.single_strip_threshold,
        clusterizer.adjacent_threshold,
    ) == (6.0, 10.0, 2.0)


def test_clusterizer_passes_thresholds_and_deposits(monkeypatch):
    plane = sc.Plane64(process_id=1, event=2, station=3)
    captured = capture_clusterizer_call(
        monkeypatch,
        sc.SimClusterizer(
            cluster_threshold=6.5,
            single_strip_threshold=11.0,
            adjacent_threshold=2.5,
            rng_seed=42,
        ),
        plane,
    )

    assert captured["Deposits"] is plane.deposits
    assert captured["EnergyThreshold_clusterStrip"] == 6.5
    assert captured["EnergyThreshold_singleStrip"] == 11.0
    assert captured["AdStripsThEnergy_singleStripCl"] == 2.5


def test_seeded_clusterizer_passes_a_smearing_generator(monkeypatch):
    captured = capture_clusterizer_call(
        monkeypatch,
        sc.SimClusterizer(rng_seed=42),
        sc.Plane64(process_id=1, event=2, station=3),
    )

    assert captured["smearing_rng"] is not None


def test_unseeded_clusterizer_leaves_the_experiment_default(monkeypatch):
    captured = capture_clusterizer_call(
        monkeypatch,
        sc.SimClusterizer(rng_seed=None),
        sc.Plane64(process_id=1, event=2, station=3),
    )

    assert captured["smearing_rng"] is None
