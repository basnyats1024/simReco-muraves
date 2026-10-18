from __future__ import annotations

import pytest

from simReco_muraves.golden_tracking.golden_root_schema import (
    infer_root_branch_type,
    infer_root_schema,
    validate_records_against_schema,
)


def valid_record():
    return {
        "ProcessID": 1,
        "EventID": 10,
        "Theta": 22.5,
        "GoldenXZGatePassed": True,
        "NonTrackClusterEnergy_Z1": [10.0, 20.0],
        "NonTrackClusterSize_Z1": [1, 2],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "bool"),
        (1, "int64"),
        (1.5, "float64"),
        ([True, False], "var * bool"),
        ([1, 2], "var * int64"),
        ([1.0, 2.0], "var * float64"),
        ([1, 2.5], "var * float64"),
    ],
)
def test_infers_supported_branch_types(value, expected):
    assert infer_root_branch_type(value) == expected


def test_infers_complete_record_schema():
    schema = infer_root_schema(valid_record())

    assert schema == {
        "ProcessID": "int64",
        "EventID": "int64",
        "Theta": "float64",
        "GoldenXZGatePassed": "bool",
        "NonTrackClusterEnergy_Z1": "var * float64",
        "NonTrackClusterSize_Z1": "var * int64",
    }


def test_preserves_branch_order():
    record = valid_record()
    schema = infer_root_schema(record)

    assert tuple(schema.keys()) == tuple(record.keys())


def test_rejects_empty_record():
    with pytest.raises(ValueError, match="empty record"):
        infer_root_schema({})


def test_rejects_empty_vector_branch():
    with pytest.raises(ValueError, match="empty vector"):
        infer_root_branch_type([])


@pytest.mark.parametrize(
    "value",
    [
        "text",
        None,
        {"value": 1},
        ["a", "b"],
        [1, "b"],
    ],
)
def test_rejects_unsupported_branch_types(value):
    with pytest.raises((TypeError, ValueError)):
        infer_root_branch_type(value)


def test_validates_records_against_schema():
    first = valid_record()
    second = valid_record()
    second["EventID"] = 11
    second["Theta"] = 23.5

    schema = infer_root_schema(first)

    validate_records_against_schema(
        [first, second],
        schema,
    )


def test_rejects_missing_record_field():
    first = valid_record()
    second = valid_record()
    del second["Theta"]

    schema = infer_root_schema(first)

    with pytest.raises(
        ValueError,
        match="Record 1 is missing fields: Theta",
    ):
        validate_records_against_schema(
            [first, second],
            schema,
        )


def test_rejects_unexpected_record_field():
    first = valid_record()
    second = valid_record()
    second["Unexpected"] = 99

    schema = infer_root_schema(first)

    with pytest.raises(
        ValueError,
        match="Record 1 has unexpected fields: Unexpected",
    ):
        validate_records_against_schema(
            [first, second],
            schema,
        )


def test_rejects_scalar_type_drift():
    first = valid_record()
    second = valid_record()
    second["Theta"] = 22

    schema = infer_root_schema(first)

    with pytest.raises(
        TypeError,
        match="branch Theta has type int64, expected float64",
    ):
        validate_records_against_schema(
            [first, second],
            schema,
        )


def test_rejects_vector_type_drift():
    first = valid_record()
    second = valid_record()
    second["NonTrackClusterSize_Z1"] = [1.0, 2.0]

    schema = infer_root_schema(first)

    with pytest.raises(
        TypeError,
        match="NonTrackClusterSize_Z1",
    ):
        validate_records_against_schema(
            [first, second],
            schema,
        )


def test_allows_empty_record_sequence():
    schema = infer_root_schema(valid_record())
    validate_records_against_schema([], schema)
