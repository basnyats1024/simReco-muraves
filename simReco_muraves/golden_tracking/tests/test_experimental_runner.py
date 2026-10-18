from __future__ import annotations

from types import SimpleNamespace

import pytest

uproot = pytest.importorskip("uproot")

from simReco_muraves.golden_tracking.experimental_compatibility import (
    build_experimental_golden_schema,
)
from simReco_muraves.golden_tracking.experimental_compatibility import (
    ExperimentalRunMetadata,
    run_experimental_golden_output,
)
from simReco_muraves.models.cluster_record import ClusterRecord


def cluster(event_id: int, station: int, energy: float):
    return ClusterRecord(
        process_id=1,
        event=event_id,
        station=station,
        cluster_id=0,
        cluster_multiplicity=1,
        cluster_energy=energy,
        cluster_position=0.0,
        cluster_size=1,
        cluster_time=100.0,
        cluster_time_spread=0.0,
        strip_ids=[0],
        strip_energies=[energy],
    )


def event_clusters(event_id: int):
    return [
        cluster(event_id, 0, 20.0),
        cluster(event_id, 1, 21.0),
        cluster(event_id, 2, 22.0),
        cluster(event_id, 3, 23.0),
        cluster(event_id, 8, 10.0),
        cluster(event_id, 9, 11.0),
        cluster(event_id, 10, 12.0),
        cluster(event_id, 11, 13.0),
    ]


def tracks():
    return SimpleNamespace(
        cluster_index_1=[0],
        cluster_index_2=[0],
        cluster_index_3=[0],
        cluster_indices_4=[[0.0]],
        ScatteringAngles=[[1.25]],
    )


def reconstruct(event, selected=True):
    return SimpleNamespace(
        process_id=event.process_id,
        event=event.event,
        tracks_xz=tracks(),
        tracks_xy=tracks(),
        best_track_xz_index=0,
        best_track_xy_index=0,
        track_3p_of_4p_index_xz=0,
        track_3p_of_4p_index_xy=0,
        track_4p_index_xz=0,
        track_4p_index_xy=0,
        best_track_3p_chi_square_xz=1.0 if selected else -1.0,
        best_track_3p_chi_square_xy=2.0,
        best_track_4p_chi_square_xz=3.0,
        best_track_4p_chi_square_xy=4.0,
        theta_4p=22.5,
        phi_4p=180.0,
    )


def metadata():
    return ExperimentalRunMetadata(
        run=123,
        trigger_rate=4.5,
        run_duration=600.0,
        working_point=54.0,
    )


def test_writes_experimental_branch_contract(tmp_path):
    output_path = tmp_path / "experimental_golden.root"

    output = run_experimental_golden_output(
        clusters=event_clusters(10),
        reconstruct_event=reconstruct,
        run_metadata=metadata(),
        output_path=output_path,
    )

    assert output.root_path == output_path
    assert output.diagnostics.selected_events == 1

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]

        assert tree.num_entries == 1
        expected_branches = set(
            build_experimental_golden_schema()
        )
        physical_branches = set(tree.keys())

        assert expected_branches <= physical_branches

        counter_branches = (
            physical_branches - expected_branches
        )

        assert len(expected_branches) == 92
        assert len(counter_branches) == 24
        assert all(
            branch_name.startswith("nNonTrackCluster")
            for branch_name in counter_branches
        )
        assert "TrackClusterExp_Z1" in tree
        assert "TrackClusterTexp_Z1" not in tree
        assert "ClusterMultriplicity_Z1" in tree
        assert "ClusterMultiplicity_Z1" not in tree
        assert "ProcessID" not in tree
        assert "EventID" not in tree


def test_writes_run_metadata_to_every_selected_entry(tmp_path):
    output_path = tmp_path / "experimental_golden.root"

    run_experimental_golden_output(
        clusters=event_clusters(20) + event_clusters(10),
        reconstruct_event=reconstruct,
        run_metadata=metadata(),
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]

        assert tree["Run"].array(library="np").tolist() == [123, 123]
        assert tree["TriggerRate"].array(library="np").tolist() == [4.5, 4.5]
        assert tree["RunDuration"].array(library="np").tolist() == [600.0, 600.0]
        assert tree["WorkingPoint"].array(library="np").tolist() == [54.0, 54.0]


def test_writes_selected_cluster_and_nontrack_values(tmp_path):
    output_path = tmp_path / "experimental_golden.root"

    run_experimental_golden_output(
        clusters=event_clusters(10),
        reconstruct_event=reconstruct,
        run_metadata=metadata(),
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]

        assert tree["TrackClusterEnergy_Z1"].array(
            library="np"
        ).tolist() == [10.0]
        assert tree["TrackClusterExp_Z1"].array(
            library="np"
        ).tolist() == [100.0]
        assert tree["ClusterMultriplicity_Z1"].array(
            library="np"
        ).tolist() == [1]
        assert tree["NonTrackClusterEnergy_Z1"].array().to_list() == [[]]


def test_writes_zero_entry_experimental_tree(tmp_path):
    output_path = tmp_path / "empty.root"

    output = run_experimental_golden_output(
        clusters=event_clusters(10),
        reconstruct_event=lambda event: reconstruct(event, selected=False),
        run_metadata=metadata(),
        output_path=output_path,
    )

    assert output.diagnostics.selected_events == 0
    assert output.diagnostics.rejected_events == 1

    with uproot.open(output_path) as root_file:
        tree = root_file["Tracks4ptree"]
        assert tree.num_entries == 0
        expected_branches = set(
            build_experimental_golden_schema()
        )
        physical_branches = set(tree.keys())

        assert expected_branches <= physical_branches

        counter_branches = (
            physical_branches - expected_branches
        )

        assert len(expected_branches) == 92
        assert len(counter_branches) == 24
        assert all(
            branch_name.startswith("nNonTrackCluster")
            for branch_name in counter_branches
        )
        assert "Run" in tree
        assert "NonTrackClusterPosition_Y4" in tree


def test_supports_custom_tree_name(tmp_path):
    output_path = tmp_path / "custom.root"

    run_experimental_golden_output(
        clusters=event_clusters(10),
        reconstruct_event=reconstruct,
        run_metadata=metadata(),
        output_path=output_path,
        tree_name="Golden4p",
    )

    with uproot.open(output_path) as root_file:
        assert "Golden4p" in root_file
        assert "Tracks4ptree" not in root_file
