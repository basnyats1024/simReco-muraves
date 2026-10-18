from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .batch_output import GoldenBatchDiagnostics, build_golden_batch
from .golden_root_schema import build_golden_root_schema, validate_record_against_golden_schema
from .root_writer import DEFAULT_TREE_NAME, write_golden_root
from .golden_pipeline import process_golden_events
from .tracking_result_adapter import build_processing_result
from ..models.tracking_event import TrackingEvent
from ..models.tracking_results import TrackingResult
from ..models.cluster_record import ClusterRecord
from ..tracking.event_builder_adapter import build_tracking_events


def build_result_index(
    results: Iterable[TrackingResult],
) -> dict[tuple[int, int], TrackingResult]:
    indexed: dict[tuple[int, int], TrackingResult] = {}

    for result in results:
        key = (result.process_id, result.event)

        if key in indexed:
            raise ValueError(
                "Duplicate TrackingResult for "
                f"ProcessID={key[0]}, EventID={key[1]}"
            )

        indexed[key] = result

    return indexed


def run_golden_tracking_output(
    *,
    events: Iterable[TrackingEvent],
    results: Iterable[TrackingResult],
    output_path: str | Path,
    tree_name: str = "Tracks4ptree",
) -> Path:
    event_list = list(events)
    result_index = build_result_index(results)

    event_keys = [
        (event.process_id, event.event)
        for event in event_list
    ]

    if len(event_keys) != len(set(event_keys)):
        raise ValueError("Duplicate TrackingEvent identity")

    missing_results = [
        key
        for key in event_keys
        if key not in result_index
    ]

    if missing_results:
        formatted = ", ".join(
            f"({process_id}, {event_id})"
            for process_id, event_id in missing_results
        )
        raise ValueError(
            "Missing TrackingResult for events: " + formatted
        )

    unexpected_results = [
        key
        for key in result_index
        if key not in set(event_keys)
    ]

    if unexpected_results:
        formatted = ", ".join(
            f"({process_id}, {event_id})"
            for process_id, event_id in unexpected_results
        )
        raise ValueError(
            "TrackingResult has no matching event: " + formatted
        )

    batch = process_golden_events(
        event_list,
        lambda event: build_processing_result(
            event,
            result_index[(event.process_id, event.event)],
        ),
    )

    records = [
        {
            branch_name: batch.columns[branch_name][entry_index]
            for branch_name in batch.columns
        }
        for entry_index in range(batch.entry_count)
    ]

    return write_golden_records(
        records=records,
        diagnostics=batch.diagnostics,
        output_path=output_path,
        tree_name=tree_name,
    )


def reconstruct_tracking_events(
    events: Iterable[TrackingEvent],
    reconstruct_event: Callable[[TrackingEvent], TrackingResult],
) -> list[TrackingResult]:
    results: list[TrackingResult] = []

    for event in events:
        result = reconstruct_event(event)

        if result.process_id != event.process_id or result.event != event.event:
            raise ValueError(
                "Reconstruction returned mismatched event identity: "
                f"expected ({event.process_id}, {event.event}), "
                f"received ({result.process_id}, {result.event})"
            )

        results.append(result)

    return results


def run_golden_from_clusters(
    *,
    clusters: Iterable[ClusterRecord],
    reconstruct_event: Callable[[TrackingEvent], TrackingResult],
    output_path: str | Path,
    tree_name: str = "Tracks4ptree",
) -> Path:
    cluster_list = list(clusters)
    events = build_tracking_events(cluster_list)
    results = reconstruct_tracking_events(events, reconstruct_event)

    return run_golden_tracking_output(
        events=events,
        results=results,
        output_path=output_path,
        tree_name=tree_name,
    )


@dataclass(frozen=True)
class GoldenOutputArtifacts:
    root_path: Path
    diagnostics_path: Path


def run_golden_reconstruction(
    *,
    clusters: Iterable[ClusterRecord],
    reconstruct_event: Callable[[TrackingEvent], TrackingResult],
    output_path: str | Path,
    diagnostics_path: str | Path | None = None,
    tree_name: str = "Tracks4ptree",
    metadata: Mapping[str, Any] | None = None,
) -> GoldenOutputArtifacts:
    events = build_tracking_events(list(clusters))

    def processor(event: TrackingEvent):
        result = reconstruct_event(event)
        return build_processing_result(event, result)

    batch = process_golden_events(events, processor)

    records = [
        {
            branch_name: batch.columns[branch_name][entry_index]
            for branch_name in batch.columns
        }
        for entry_index in range(batch.entry_count)
    ]

    root_path = write_golden_records(
        records=records,
        diagnostics=batch.diagnostics,
        output_path=output_path,
        tree_name=tree_name,
    )

    if diagnostics_path is None:
        root_path_object = Path(root_path)
        diagnostics_path = root_path_object.with_suffix(
            root_path_object.suffix + ".diagnostics.json"
        )

    diagnostics_metadata = dict(metadata or {})
    diagnostics_metadata.update({
        "tree_name": tree_name,
        "root_output": str(root_path),
    })

    written_diagnostics_path = write_diagnostics_json(
        diagnostics=batch.diagnostics,
        output_path=diagnostics_path,
        metadata=diagnostics_metadata,
    )

    return GoldenOutputArtifacts(
        root_path=Path(root_path),
        diagnostics_path=written_diagnostics_path,
    )
def diagnostics_to_dict(
    diagnostics: GoldenBatchDiagnostics,
) -> dict[str, Any]:
    diagnostics.validate_totals()

    data = asdict(diagnostics)

    if diagnostics.input_events == 0:
        data["selection_efficiency"] = 0.0
        data["rejection_fraction"] = 0.0
        data["consistency_error_fraction"] = 0.0
    else:
        denominator = float(diagnostics.input_events)
        data["selection_efficiency"] = (
            diagnostics.selected_events / denominator
        )
        data["rejection_fraction"] = (
            diagnostics.rejected_events / denominator
        )
        data["consistency_error_fraction"] = (
            diagnostics.consistency_errors / denominator
        )

    return data


def write_diagnostics_json(
    *,
    diagnostics: GoldenBatchDiagnostics,
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output_path)

    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Diagnostics output path is a directory: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "diagnostics": diagnostics_to_dict(diagnostics),
        "metadata": dict(metadata or {}),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    return path
def write_golden_records(
    *,
    records: Iterable[Mapping[str, Any]],
    diagnostics: GoldenBatchDiagnostics,
    output_path: str | Path,
    tree_name: str = DEFAULT_TREE_NAME,
) -> Path:
    copied_records = [dict(record) for record in records]

    for record in copied_records:
        validate_record_against_golden_schema(record)

    batch = build_golden_batch(
        selected_records=copied_records,
        diagnostics=diagnostics,
    )

    return write_golden_root(
        batch=batch,
        output_path=output_path,
        tree_name=tree_name,
        empty_branch_types=build_golden_root_schema(),
    )
