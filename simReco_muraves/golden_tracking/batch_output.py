from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class GoldenBatchDiagnostics:
    input_events: int = 0
    rejected_events: int = 0
    selected_events: int = 0
    consistency_errors: int = 0

    rejection_counts: dict[str, int] = field(
        default_factory=dict
    )
    consistency_error_counts: dict[str, int] = field(
        default_factory=dict
    )

    def record_input(self) -> None:
        self.input_events += 1

    def record_selected(self) -> None:
        self.selected_events += 1

    def record_rejection(self, reason: str) -> None:
        self.rejected_events += 1
        self.rejection_counts[reason] = (
            self.rejection_counts.get(reason, 0) + 1
        )

    def record_consistency_error(self, reason: str) -> None:
        self.consistency_errors += 1
        self.consistency_error_counts[reason] = (
            self.consistency_error_counts.get(reason, 0) + 1
        )

    def validate_totals(self) -> None:
        classified = (
            self.selected_events
            + self.rejected_events
            + self.consistency_errors
        )

        if classified != self.input_events:
            raise ValueError(
                "Diagnostic totals are inconsistent: "
                f"input_events={self.input_events}, "
                f"classified_events={classified}"
            )


@dataclass(frozen=True)
class GoldenBatch:
    columns: dict[str, list[Any]]
    diagnostics: GoldenBatchDiagnostics

    @property
    def entry_count(self) -> int:
        if not self.columns:
            return 0

        first_column = next(iter(self.columns.values()))
        return len(first_column)


def _copy_branch_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)

    if isinstance(value, tuple):
        return list(value)

    return value


def build_columnar_batch(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    if not records:
        return {}

    expected_fields = tuple(records[0].keys())
    expected_field_set = set(expected_fields)

    columns: dict[str, list[Any]] = {
        field_name: []
        for field_name in expected_fields
    }

    for record_index, record in enumerate(records):
        record_field_set = set(record.keys())

        missing_fields = [
            field_name
            for field_name in expected_fields
            if field_name not in record
        ]

        unexpected_fields = [
            field_name
            for field_name in record
            if field_name not in expected_field_set
        ]

        if missing_fields:
            raise ValueError(
                f"Record {record_index} is missing fields: "
                + ", ".join(missing_fields)
            )

        if unexpected_fields:
            raise ValueError(
                f"Record {record_index} has unexpected fields: "
                + ", ".join(unexpected_fields)
            )

        for field_name in expected_fields:
            columns[field_name].append(
                _copy_branch_value(record[field_name])
            )

    validate_column_lengths(columns)

    return columns


def validate_column_lengths(
    columns: Mapping[str, Sequence[Any]],
) -> int:
    if not columns:
        return 0

    lengths = {
        field_name: len(values)
        for field_name, values in columns.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        details = ", ".join(
            f"{field_name}={length}"
            for field_name, length in lengths.items()
        )

        raise ValueError(
            "Output columns have inconsistent lengths: "
            + details
        )

    return next(iter(unique_lengths))


def build_golden_batch(
    *,
    selected_records: Sequence[Mapping[str, Any]],
    diagnostics: GoldenBatchDiagnostics,
) -> GoldenBatch:
    diagnostics.validate_totals()

    columns = build_columnar_batch(selected_records)
    entry_count = validate_column_lengths(columns)

    if entry_count != diagnostics.selected_events:
        raise ValueError(
            "Selected record count does not match diagnostics: "
            f"records={entry_count}, "
            f"selected_events={diagnostics.selected_events}"
        )

    return GoldenBatch(
        columns=columns,
        diagnostics=diagnostics,
    )
