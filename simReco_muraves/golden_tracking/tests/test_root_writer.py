from __future__ import annotations

from pathlib import Path

import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("awkward")
pytest.importorskip("numpy")

from simReco_muraves.golden_tracking.batch_output import (
    GoldenBatch,
    GoldenBatchDiagnostics,
    build_golden_batch,
)
from simReco_muraves.golden_tracking.root_writer import (
    DEFAULT_TREE_NAME,
    write_golden_root,
)


def make_record(process_id: int, event_id: int) -> dict:
    return {
        "ProcessID": process_id,
        "EventID": event_id,
        "Theta": 22.5 + event_id,
        "Phi": 180.0,
        "ClusterMultiplicity_Z1": 3,
        "NonTrackClusterEnergy_Z1": [10.0, 20.0],
        "NonTrackClusterSize_Z1": [1, 2],
    }


def make_selected_batch():
    diagnostics = GoldenBatchDiagnostics()
    diagnostics.record_input()
    diagnostics.record_selected()
    diagnostics.record_input()
    diagnostics.record_selected()

    return build_golden_batch(
        selected_records=[
            make_record(1, 10),
            make_record(1, 11),
        ],
        diagnostics=diagnostics,
    )


def make_empty_batch():
    diagnostics = GoldenBatchDiagnostics()
    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")
    diagnostics.record_input()
    diagnostics.record_rejection("invalid_4p_xz")

    return build_golden_batch(
        selected_records=[],
        diagnostics=diagnostics,
    )


def empty_schema():
    return {
        "ProcessID": "int64",
        "EventID": "int64",
        "Theta": "float64",
        "Phi": "float64",
        "ClusterMultiplicity_Z1": "int64",
    }


def test_writes_tracks4p_tree(tmp_path):
    output_path = tmp_path / "golden.root"
    write_golden_root(batch=make_selected_batch(), output_path=output_path)

    assert output_path.exists()
    with uproot.open(output_path) as root_file:
        assert DEFAULT_TREE_NAME in root_file
        assert root_file[DEFAULT_TREE_NAME].num_entries == 2


def test_writes_scalar_branches(tmp_path):
    output_path = tmp_path / "golden.root"
    write_golden_root(batch=make_selected_batch(), output_path=output_path)

    with uproot.open(output_path) as root_file:
        tree = root_file[DEFAULT_TREE_NAME]
        assert tree["ProcessID"].array(library="np").tolist() == [1, 1]
        assert tree["EventID"].array(library="np").tolist() == [10, 11]
        assert tree["Theta"].array(library="np").tolist() == [32.5, 33.5]
        assert tree["ClusterMultiplicity_Z1"].array(library="np").tolist() == [3, 3]


def test_writes_jagged_vector_branches(tmp_path):
    output_path = tmp_path / "golden.root"
    write_golden_root(batch=make_selected_batch(), output_path=output_path)

    with uproot.open(output_path) as root_file:
        tree = root_file[DEFAULT_TREE_NAME]
        energies = tree["NonTrackClusterEnergy_Z1"].array().to_list()
        sizes = tree["NonTrackClusterSize_Z1"].array().to_list()
        assert energies == [[10.0, 20.0], [10.0, 20.0]]
        assert sizes == [[1, 2], [1, 2]]


def test_supports_custom_tree_name(tmp_path):
    output_path = tmp_path / "custom.root"
    write_golden_root(
        batch=make_selected_batch(),
        output_path=output_path,
        tree_name="GoldenTracks",
    )

    with uproot.open(output_path) as root_file:
        assert "GoldenTracks" in root_file
        assert DEFAULT_TREE_NAME not in root_file


def test_creates_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "directory" / "golden.root"
    write_golden_root(batch=make_selected_batch(), output_path=output_path)
    assert output_path.exists()


def test_overwrites_existing_root_file(tmp_path):
    output_path = tmp_path / "golden.root"
    output_path.write_bytes(b"not a ROOT file")
    write_golden_root(batch=make_selected_batch(), output_path=output_path)

    with uproot.open(output_path) as root_file:
        assert DEFAULT_TREE_NAME in root_file


def test_rejects_empty_tree_name(tmp_path):
    with pytest.raises(ValueError, match="non-empty string"):
        write_golden_root(
            batch=make_selected_batch(),
            output_path=tmp_path / "golden.root",
            tree_name="",
        )


def test_rejects_directory_as_output_path(tmp_path):
    with pytest.raises(IsADirectoryError):
        write_golden_root(batch=make_selected_batch(), output_path=tmp_path)


def test_writes_valid_zero_entry_root_file(tmp_path):
    output_path = tmp_path / "empty_golden.root"
    write_golden_root(
        batch=make_empty_batch(),
        output_path=output_path,
        empty_branch_types=empty_schema(),
    )

    with uproot.open(output_path) as root_file:
        tree = root_file[DEFAULT_TREE_NAME]
        assert tree.num_entries == 0
        assert "ProcessID" in tree
        assert "EventID" in tree
        assert "Theta" in tree
        assert "Phi" in tree
        assert "ClusterMultiplicity_Z1" in tree


def test_zero_entry_output_requires_schema(tmp_path):
    with pytest.raises(ValueError, match="empty_branch_types are required"):
        write_golden_root(
            batch=make_empty_batch(),
            output_path=tmp_path / "empty.root",
        )


def test_rejects_selected_count_mismatch(tmp_path):
    diagnostics = GoldenBatchDiagnostics(input_events=1, selected_events=1)
    batch = GoldenBatch(columns={}, diagnostics=diagnostics)

    with pytest.raises(ValueError, match="selected-event count"):
        write_golden_root(
            batch=batch,
            output_path=tmp_path / "invalid.root",
            empty_branch_types={"EventID": "int64"},
        )


def test_returns_written_path(tmp_path):
    output_path = tmp_path / "golden.root"
    result = write_golden_root(
        batch=make_selected_batch(),
        output_path=output_path,
    )
    assert result == Path(output_path)
