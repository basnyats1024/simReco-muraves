from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .batch_output import (
    GoldenBatch,
    GoldenBatchDiagnostics,
    build_golden_batch,
)


@dataclass(frozen=True)
class GoldenProcessingResult:
    selected: bool
    record: Mapping[str, Any] | None = None
    rejection_reason: str | None = None
    consistency_error: str | None = None

    def __post_init__(self) -> None:
        states = sum(
            value is not None
            for value in (
                self.record,
                self.rejection_reason,
                self.consistency_error,
            )
        )

        if states != 1:
            raise ValueError(
                "Exactly one processing outcome must be provided"
            )

        if self.selected and self.record is None:
            raise ValueError(
                "A selected result must contain a record"
            )

        if not self.selected and self.record is not None:
            raise ValueError(
                "A rejected result cannot contain a record"
            )


def selected_result(
    record: Mapping[str, Any],
) -> GoldenProcessingResult:
    return GoldenProcessingResult(
        selected=True,
        record=record,
    )


def rejected_result(
    reason: str,
) -> GoldenProcessingResult:
    if not reason:
        raise ValueError("Rejection reason must be non-empty")

    return GoldenProcessingResult(
        selected=False,
        rejection_reason=reason,
    )


def consistency_error_result(
    reason: str,
) -> GoldenProcessingResult:
    if not reason:
        raise ValueError(
            "Consistency-error reason must be non-empty"
        )

    return GoldenProcessingResult(
        selected=False,
        consistency_error=reason,
    )


def process_golden_events(
    events: Iterable[Any],
    processor: Callable[[Any], GoldenProcessingResult],
) -> GoldenBatch:
    diagnostics = GoldenBatchDiagnostics()
    selected_records: list[Mapping[str, Any]] = []

    for event in events:
        diagnostics.record_input()
        result = processor(event)

        required_attributes = (
            "selected",
            "record",
            "rejection_reason",
            "consistency_error",
        )

        if not all(
            hasattr(result, attribute)
            for attribute in required_attributes
        ):
            raise TypeError(
                "processor must return GoldenProcessingResult"
            )

        if result.selected:
            diagnostics.record_selected()
            assert result.record is not None
            selected_records.append(dict(result.record))
            continue

        if result.consistency_error is not None:
            diagnostics.record_consistency_error(
                result.consistency_error
            )
            continue

        assert result.rejection_reason is not None
        diagnostics.record_rejection(
            result.rejection_reason
        )

    return build_golden_batch(
        selected_records=selected_records,
        diagnostics=diagnostics,
    )
