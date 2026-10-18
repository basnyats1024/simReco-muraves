from __future__ import annotations

import pytest

from simReco_muraves.golden_tracking.golden_pipeline import (
    GoldenProcessingResult,
    consistency_error_result,
    process_golden_events,
    rejected_result,
    selected_result,
)


def test_builds_batch_from_selected_events():
    events = [1, 2, 3]

    def processor(event):
        return selected_result({
            "EventID": event,
            "Theta": float(event),
        })

    batch = process_golden_events(events, processor)

    assert batch.entry_count == 3
    assert batch.columns["EventID"] == [1, 2, 3]
    assert batch.columns["Theta"] == [1.0, 2.0, 3.0]
    assert batch.diagnostics.input_events == 3
    assert batch.diagnostics.selected_events == 3


def test_records_normal_rejections():
    events = [1, 2, 3]

    def processor(event):
        if event == 1:
            return selected_result({"EventID": event})
        return rejected_result("invalid_3p_xz")

    batch = process_golden_events(events, processor)

    assert batch.entry_count == 1
    assert batch.diagnostics.rejected_events == 2
    assert batch.diagnostics.rejection_counts == {
        "invalid_3p_xz": 2,
    }


def test_records_consistency_errors_separately():
    events = [1, 2, 3]

    def processor(event):
        if event == 1:
            return selected_result({"EventID": event})
        if event == 2:
            return rejected_result("invalid_4p_xz")
        return consistency_error_result("invalid_xy_state")

    batch = process_golden_events(events, processor)

    assert batch.entry_count == 1
    assert batch.diagnostics.selected_events == 1
    assert batch.diagnostics.rejected_events == 1
    assert batch.diagnostics.consistency_errors == 1
    assert batch.diagnostics.rejection_counts == {
        "invalid_4p_xz": 1,
    }
    assert batch.diagnostics.consistency_error_counts == {
        "invalid_xy_state": 1,
    }


def test_all_rejected_events_produce_valid_zero_entry_batch():
    events = [1, 2]

    def processor(event):
        return rejected_result("invalid_3p_xz")

    batch = process_golden_events(events, processor)

    assert batch.entry_count == 0
    assert batch.columns == {}
    assert batch.diagnostics.input_events == 2
    assert batch.diagnostics.selected_events == 0
    assert batch.diagnostics.rejected_events == 2


def test_input_order_is_preserved():
    events = [30, 10, 20]

    def processor(event):
        return selected_result({"EventID": event})

    batch = process_golden_events(events, processor)

    assert batch.columns["EventID"] == [30, 10, 20]


def test_selected_record_is_copied():
    source_record = {
        "EventID": 1,
        "Values": [10.0, 20.0],
    }

    batch = process_golden_events(
        [source_record],
        lambda event: selected_result(event),
    )

    source_record["EventID"] = 99
    source_record["Values"].append(30.0)

    assert batch.columns["EventID"] == [1]
    assert batch.columns["Values"] == [[10.0, 20.0]]


def test_rejects_invalid_processor_return_type():
    with pytest.raises(
        TypeError,
        match="GoldenProcessingResult",
    ):
        process_golden_events(
            [1],
            lambda event: {"EventID": event},
        )


def test_selected_result_requires_record():
    with pytest.raises(
        ValueError,
        match="selected result must contain a record",
    ):
        GoldenProcessingResult(
            selected=True,
            rejection_reason="invalid",
        )


def test_rejected_result_cannot_contain_record():
    with pytest.raises(
        ValueError,
        match="rejected result cannot contain a record",
    ):
        GoldenProcessingResult(
            selected=False,
            record={"EventID": 1},
        )


def test_requires_exactly_one_outcome():
    with pytest.raises(
        ValueError,
        match="Exactly one processing outcome",
    ):
        GoldenProcessingResult(selected=False)

    with pytest.raises(
        ValueError,
        match="Exactly one processing outcome",
    ):
        GoldenProcessingResult(
            selected=False,
            rejection_reason="invalid_3p_xz",
            consistency_error="invalid_xy_state",
        )


@pytest.mark.parametrize(
    "factory",
    [
        rejected_result,
        consistency_error_result,
    ],
)
def test_reasons_must_be_nonempty(factory):
    with pytest.raises(ValueError, match="non-empty"):
        factory("")
