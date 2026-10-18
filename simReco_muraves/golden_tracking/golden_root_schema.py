from __future__ import annotations

from typing import Any, Mapping, Sequence

from .event_cluster_output import PLANE_NAMES
from .final_output import (
    REQUIRED_FLAG_FIELDS,
    REQUIRED_INDEX_FIELDS,
    REQUIRED_TRACK_FIELDS,
)


def build_golden_root_schema() -> dict[str, str]:
    schema: dict[str, str] = {
        "ProcessID": "int64",
        "EventID": "int64",
    }

    for field_name in REQUIRED_TRACK_FIELDS:
        schema[field_name] = "float64"

    for field_name in REQUIRED_INDEX_FIELDS:
        schema[field_name] = "int64"

    # Must stay in the same position as in the record builder: the
    # schema test compares key order, not just key membership.
    for field_name in REQUIRED_FLAG_FIELDS:
        schema[field_name] = "int64"

    for plane_name in PLANE_NAMES:
        suffix = f"{plane_name[0].upper()}{plane_name[1]}"

        schema[f"TrackClusterIndex_{suffix}"] = "int64"
        schema[f"TrackClusterEnergy_{suffix}"] = "float64"
        schema[f"TrackClusterPosition_{suffix}"] = "float64"
        schema[f"TrackClusterSize_{suffix}"] = "int64"
        schema[f"TrackClusterTexp_{suffix}"] = "float64"
        schema[f"TrackClusterMaxStripEnergy_{suffix}"] = "float64"
        schema[f"TrackClusterMinStripEnergy_{suffix}"] = "float64"
        schema[f"ClusterMultiplicity_{suffix}"] = "int64"
        schema[f"NonTrackClusterEnergy_{suffix}"] = "var * float64"
        schema[f"NonTrackClusterPosition_{suffix}"] = "var * float64"
        schema[f"NonTrackClusterSize_{suffix}"] = "var * int64"

    return schema


def validate_record_against_golden_schema(
    record: Mapping[str, object],
) -> None:
    schema = build_golden_root_schema()

    missing = [
        branch_name
        for branch_name in schema
        if branch_name not in record
    ]

    unexpected = [
        branch_name
        for branch_name in record
        if branch_name not in schema
    ]

    if missing:
        raise ValueError(
            "Golden record is missing branches: "
            + ", ".join(missing)
        )

    if unexpected:
        raise ValueError(
            "Golden record has unexpected branches: "
            + ", ".join(unexpected)
        )
SCALAR_TYPE_MAP = {
    bool: "bool",
    int: "int64",
    float: "float64",
}


def infer_root_branch_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int64"

    if isinstance(value, float):
        return "float64"

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(
                "Cannot infer the type of an empty vector branch"
            )

        element_types = {type(element) for element in value}

        if element_types == {bool}:
            return "var * bool"

        if element_types <= {int}:
            return "var * int64"

        if element_types <= {int, float}:
            return "var * float64"

        raise TypeError(
            "Unsupported or mixed vector branch element types"
        )

    raise TypeError(
        "Unsupported ROOT branch value type: "
        f"{type(value).__name__}"
    )


def infer_root_schema(
    record: Mapping[str, Any],
) -> dict[str, str]:
    if not record:
        raise ValueError(
            "Cannot infer a ROOT schema from an empty record"
        )

    return {
        branch_name: infer_root_branch_type(value)
        for branch_name, value in record.items()
    }


def validate_records_against_schema(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, str],
) -> None:
    expected_fields = tuple(schema.keys())
    expected_field_set = set(expected_fields)

    for record_index, record in enumerate(records):
        record_field_set = set(record.keys())

        missing = [
            field
            for field in expected_fields
            if field not in record_field_set
        ]

        unexpected = [
            field
            for field in record
            if field not in expected_field_set
        ]

        if missing:
            raise ValueError(
                f"Record {record_index} is missing fields: "
                + ", ".join(missing)
            )

        if unexpected:
            raise ValueError(
                f"Record {record_index} has unexpected fields: "
                + ", ".join(unexpected)
            )

        for branch_name, expected_type in schema.items():
            actual_type = infer_root_branch_type(
                record[branch_name]
            )

            if actual_type != expected_type:
                raise TypeError(
                    f"Record {record_index} branch {branch_name} "
                    f"has type {actual_type}, expected {expected_type}"
                )
