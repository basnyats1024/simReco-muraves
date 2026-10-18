from __future__ import annotations

import json
from pathlib import Path

import pytest

from simReco_muraves.golden_tracking.batch_output import GoldenBatchDiagnostics
from simReco_muraves.golden_tracking.production_pipeline import (
    diagnostics_to_dict,
    write_diagnostics_json,
)


def populated_diagnostics():
    diagnostics = GoldenBatchDiagnostics()

    diagnostics.record_input()
    diagnostics.record_selected()

    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")

    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")

    diagnostics.record_input()
    diagnostics.record_consistency_error("invalid_xy_state")

    return diagnostics


def test_converts_diagnostics_to_serializable_dictionary():
    data = diagnostics_to_dict(populated_diagnostics())

    assert data["input_events"] == 4
    assert data["selected_events"] == 1
    assert data["rejected_events"] == 2
    assert data["consistency_errors"] == 1
    assert data["rejection_counts"] == {
        "invalid_3p_xz": 2,
    }
    assert data["consistency_error_counts"] == {
        "invalid_xy_state": 1,
    }


def test_calculates_selection_fractions():
    data = diagnostics_to_dict(populated_diagnostics())

    assert data["selection_efficiency"] == 0.25
    assert data["rejection_fraction"] == 0.5
    assert data["consistency_error_fraction"] == 0.25


def test_zero_input_has_zero_fractions():
    data = diagnostics_to_dict(GoldenBatchDiagnostics())

    assert data["selection_efficiency"] == 0.0
    assert data["rejection_fraction"] == 0.0
    assert data["consistency_error_fraction"] == 0.0


def test_rejects_inconsistent_totals():
    diagnostics = GoldenBatchDiagnostics(
        input_events=2,
        selected_events=1,
    )

    with pytest.raises(ValueError, match="totals are inconsistent"):
        diagnostics_to_dict(diagnostics)


def test_writes_diagnostics_json(tmp_path):
    output_path = tmp_path / "golden_diagnostics.json"

    result = write_diagnostics_json(
        diagnostics=populated_diagnostics(),
        output_path=output_path,
        metadata={
            "tree_name": "Tracks4ptree",
            "sample": "test_mc",
        },
    )

    assert result == Path(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["diagnostics"]["input_events"] == 4
    assert payload["diagnostics"]["selected_events"] == 1
    assert payload["metadata"]["tree_name"] == "Tracks4ptree"
    assert payload["metadata"]["sample"] == "test_mc"


def test_writes_empty_metadata_by_default(tmp_path):
    output_path = tmp_path / "golden_diagnostics.json"

    write_diagnostics_json(
        diagnostics=GoldenBatchDiagnostics(),
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"] == {}


def test_creates_parent_directories(tmp_path):
    output_path = tmp_path / "nested" / "golden_diagnostics.json"

    write_diagnostics_json(
        diagnostics=GoldenBatchDiagnostics(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_rejects_directory_output_path(tmp_path):
    with pytest.raises(IsADirectoryError):
        write_diagnostics_json(
            diagnostics=GoldenBatchDiagnostics(),
            output_path=tmp_path,
        )
