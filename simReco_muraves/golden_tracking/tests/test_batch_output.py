from __future__ import annotations

import pytest

from simReco_muraves.golden_tracking.batch_output import (
    GoldenBatchDiagnostics,
    build_columnar_batch,
    build_golden_batch,
    validate_column_lengths,
)


def make_record(
    process_id: int,
    event_id: int,
) -> dict:
    return {
        "ProcessID": process_id,
        "EventID": event_id,
        "Theta": 20.0 + event_id,
        "ClusterMultiplicity_Z1": 2,
        "NonTrackClusterEnergy_Z1": [
            10.0 + event_id,
            20.0 + event_id,
        ],
    }


def test_builds_columnar_batch():
    records = [
        make_record(1, 10),
        make_record(1, 11),
        make_record(2, 12),
    ]

    columns = build_columnar_batch(records)

    assert columns["ProcessID"] == [1, 1, 2]
    assert columns["EventID"] == [10, 11, 12]
    assert columns["Theta"] == [30.0, 31.0, 32.0]
    assert columns["ClusterMultiplicity_Z1"] == [2, 2, 2]


def test_preserves_branch_order():
    records = [
        make_record(1, 10),
        make_record(1, 11),
    ]

    columns = build_columnar_batch(records)

    assert tuple(columns.keys()) == tuple(records[0].keys())


def test_vector_values_are_copied():
    records = [
        make_record(1, 10),
        make_record(1, 11),
    ]

    columns = build_columnar_batch(records)

    columns["NonTrackClusterEnergy_Z1"][0].append(999.0)

    assert records[0]["NonTrackClusterEnergy_Z1"] == [
        20.0,
        30.0,
    ]


def test_vector_entries_are_independent():
    shared = [10.0, 20.0]

    records = [
        {
            "EventID": 1,
            "Values": shared,
        },
        {
            "EventID": 2,
            "Values": shared,
        },
    ]

    columns = build_columnar_batch(records)

    columns["Values"][0].append(999.0)

    assert columns["Values"][1] == [10.0, 20.0]
    assert shared == [10.0, 20.0]


def test_empty_record_collection_returns_empty_columns():
    columns = build_columnar_batch([])

    assert columns == {}


def test_rejects_missing_record_field():
    records = [
        make_record(1, 10),
        make_record(1, 11),
    ]

    del records[1]["Theta"]

    with pytest.raises(
        ValueError,
        match="Record 1 is missing fields: Theta",
    ):
        build_columnar_batch(records)


def test_rejects_unexpected_record_field():
    records = [
        make_record(1, 10),
        make_record(1, 11),
    ]

    records[1]["Unexpected"] = 99

    with pytest.raises(
        ValueError,
        match="Record 1 has unexpected fields: Unexpected",
    ):
        build_columnar_batch(records)


def test_validates_equal_column_lengths():
    columns = {
        "EventID": [1, 2, 3],
        "Theta": [10.0, 20.0, 30.0],
    }

    assert validate_column_lengths(columns) == 3


def test_rejects_inconsistent_column_lengths():
    columns = {
        "EventID": [1, 2, 3],
        "Theta": [10.0, 20.0],
    }

    with pytest.raises(
        ValueError,
        match="inconsistent lengths",
    ):
        validate_column_lengths(columns)


def test_empty_columns_have_zero_entries():
    assert validate_column_lengths({}) == 0


def test_records_diagnostic_counts():
    diagnostics = GoldenBatchDiagnostics()

    diagnostics.record_input()
    diagnostics.record_selected()

    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")

    diagnostics.record_input()
    diagnostics.record_consistency_error(
        "invalid_xy_state"
    )

    assert diagnostics.input_events == 3
    assert diagnostics.selected_events == 1
    assert diagnostics.rejected_events == 1
    assert diagnostics.consistency_errors == 1

    assert diagnostics.rejection_counts == {
        "invalid_3p_xz": 1,
    }

    assert diagnostics.consistency_error_counts == {
        "invalid_xy_state": 1,
    }


def test_accumulates_repeated_diagnostic_reasons():
    diagnostics = GoldenBatchDiagnostics()

    for _ in range(3):
        diagnostics.record_input()
        diagnostics.record_rejection("invalid_4p_xz")

    diagnostics.validate_totals()

    assert diagnostics.rejection_counts == {
        "invalid_4p_xz": 3,
    }


def test_rejects_inconsistent_diagnostic_totals():
    diagnostics = GoldenBatchDiagnostics(
        input_events=5,
        selected_events=2,
        rejected_events=1,
        consistency_errors=1,
    )

    with pytest.raises(
        ValueError,
        match="Diagnostic totals are inconsistent",
    ):
        diagnostics.validate_totals()


def test_builds_complete_golden_batch():
    records = [
        make_record(1, 10),
        make_record(1, 11),
    ]

    diagnostics = GoldenBatchDiagnostics()

    diagnostics.record_input()
    diagnostics.record_selected()

    diagnostics.record_input()
    diagnostics.record_selected()

    batch = build_golden_batch(
        selected_records=records,
        diagnostics=diagnostics,
    )

    assert batch.entry_count == 2
    assert batch.columns["EventID"] == [10, 11]
    assert batch.diagnostics.selected_events == 2


def test_allows_valid_zero_entry_batch():
    diagnostics = GoldenBatchDiagnostics()

    diagnostics.record_input()
    diagnostics.record_rejection("invalid_3p_xz")

    diagnostics.record_input()
    diagnostics.record_rejection("invalid_4p_xz")

    batch = build_golden_batch(
        selected_records=[],
        diagnostics=diagnostics,
    )

    assert batch.entry_count == 0
    assert batch.columns == {}
    assert batch.diagnostics.input_events == 2
    assert batch.diagnostics.selected_events == 0


def test_rejects_selected_record_count_mismatch():
    diagnostics = GoldenBatchDiagnostics(
        input_events=3,
        selected_events=2,
        rejected_events=1,
    )

    with pytest.raises(
        ValueError,
        match="Selected record count does not match diagnostics",
    ):
        build_golden_batch(
            selected_records=[make_record(1, 10)],
            diagnostics=diagnostics,
        )
