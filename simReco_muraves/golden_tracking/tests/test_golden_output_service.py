from __future__ import annotations

import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("awkward")
pytest.importorskip("numpy")

from simReco_muraves.golden_tracking.batch_output import GoldenBatchDiagnostics
from simReco_muraves.golden_tracking.event_cluster_output import (
    PlaneClusterInput,
    build_golden_event_cluster_output,
)
from simReco_muraves.golden_tracking.final_output import (
    GoldenEventIdentity,
    build_final_golden_branches,
)
from simReco_muraves.golden_tracking.production_pipeline import write_golden_records
from simReco_muraves.golden_tracking.root_writer import DEFAULT_TREE_NAME
from simReco_muraves.golden_tracking.event_cluster_output import SelectedClusterIndices


def make_plane(base: float, count: int) -> PlaneClusterInput:
    energies = [base + 10.0 * index for index in range(count)]
    positions = [base + float(index) for index in range(count)]
    sizes = [index + 1 for index in range(count)]
    texp = [100.0 + base + float(index) for index in range(count)]
    strip_energies = [
        [
            energies[index] - float(strip_index)
            for strip_index in range(sizes[index])
        ]
        for index in range(count)
    ]

    return PlaneClusterInput(
        energies=energies,
        positions=positions,
        sizes=sizes,
        texp=texp,
        strip_energies=strip_energies,
    )


def make_record(event_id: int):
    planes = {
        "z1": make_plane(10.0, 3),
        "z2": make_plane(20.0, 4),
        "z3": make_plane(30.0, 5),
        "z4": make_plane(40.0, 4),
        "y1": make_plane(50.0, 2),
        "y2": make_plane(60.0, 3),
        "y3": make_plane(70.0, 4),
        "y4": make_plane(80.0, 3),
    }

    selected = SelectedClusterIndices(
        z1=2,
        z2=3,
        z3=4,
        z4=3,
        y1=1,
        y2=2,
        y3=3,
        y4=2,
    )

    cluster_output = build_golden_event_cluster_output(
        selected=selected,
        planes=planes,
    )

    return build_final_golden_branches(
        identity=GoldenEventIdentity(
            process_id=1,
            event_id=event_id,
        ),
        track_fields={
            "Theta": 22.5,
            "Phi": 180.0,
            "Scattering_xz": 1.0,
            "Scattering_xy": -1.0,
            "ChiSquare_xz": 3.0,
            "ChiSquare_xy": 4.0,
            "ChiSquare_3p_xz": 1.0,
            "ChiSquare_3p_xy": 2.0,
        },
        selected_index_fields={
            "BestTrack3pIndex_xz": 1,
            "BestTrack3pIndex_xy": 0,
            "Track3pOf4pRow_xz": 1,
            "Track3pOf4pRow_xy": 0,
            "Track4pInnerIndex_xz": 0,
            "Track4pInnerIndex_xy": 0,
        },
        cluster_output=cluster_output,
    )


def selected_diagnostics(count: int):
    diagnostics = GoldenBatchDiagnostics()
    for _ in range(count):
        diagnostics.record_input()
        diagnostics.record_selected()
    return diagnostics


def test_writes_complete_records(tmp_path):
    output_path = tmp_path / "golden.root"

    result = write_golden_records(
        records=[make_record(10), make_record(11)],
        diagnostics=selected_diagnostics(2),
        output_path=output_path,
    )

    assert result == output_path

    with uproot.open(output_path) as root_file:
        tree = root_file[DEFAULT_TREE_NAME]
        assert tree.num_entries == 2
        assert tree["EventID"].array(library="np").tolist() == [10, 11]
        assert "ClusterMultiplicity_Z1" in tree
        assert "ClusterMultriplicity_Z1" not in tree


def test_writes_zero_entry_file_with_full_schema(tmp_path):
    diagnostics = GoldenBatchDiagnostics()
    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")

    output_path = tmp_path / "empty.root"

    write_golden_records(
        records=[],
        diagnostics=diagnostics,
        output_path=output_path,
    )

    with uproot.open(output_path) as root_file:
        tree = root_file[DEFAULT_TREE_NAME]
        assert tree.num_entries == 0
        assert "ProcessID" in tree
        assert "TrackClusterEnergy_Z1" in tree
        assert "NonTrackClusterEnergy_Y4" in tree


def test_supports_custom_tree_name(tmp_path):
    output_path = tmp_path / "custom.root"

    write_golden_records(
        records=[make_record(10)],
        diagnostics=selected_diagnostics(1),
        output_path=output_path,
        tree_name="GoldenTracks",
    )

    with uproot.open(output_path) as root_file:
        assert "GoldenTracks" in root_file
        assert DEFAULT_TREE_NAME not in root_file


def test_rejects_record_missing_schema_branch(tmp_path):
    record = make_record(10)
    del record["Theta"]

    with pytest.raises(ValueError, match="missing branches: Theta"):
        write_golden_records(
            records=[record],
            diagnostics=selected_diagnostics(1),
            output_path=tmp_path / "invalid.root",
        )


def test_rejects_record_with_unexpected_branch(tmp_path):
    record = make_record(10)
    record["Unexpected"] = 1

    with pytest.raises(ValueError, match="unexpected branches: Unexpected"):
        write_golden_records(
            records=[record],
            diagnostics=selected_diagnostics(1),
            output_path=tmp_path / "invalid.root",
        )


def test_rejects_record_count_mismatch(tmp_path):
    with pytest.raises(
        ValueError,
        match="Selected record count does not match diagnostics",
    ):
        write_golden_records(
            records=[make_record(10)],
            diagnostics=selected_diagnostics(2),
            output_path=tmp_path / "invalid.root",
        )
