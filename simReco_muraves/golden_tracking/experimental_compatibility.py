from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .batch_output import GoldenBatchDiagnostics, build_golden_batch
from .golden_pipeline import process_golden_events
from .root_writer import write_golden_root
from .tracking_result_adapter import build_processing_result
from ..models.cluster_record import ClusterRecord
from ..models.tracking_event import TrackingEvent
from ..models.tracking_results import TrackingResult
from ..tracking.event_builder_adapter import build_tracking_events


PLANE_SUFFIXES = (
    "Z1", "Z2", "Z3", "Z4",
    "Y1", "Y2", "Y3", "Y4",
)

TRACK_BRANCHES = (
    "Phi",
    "Theta",
    "Scattering_xy",
    "Scattering_xz",
    "ChiSquare_xz",
    "ChiSquare_xy",
    "ChiSquare_3p_xz",
    "ChiSquare_3p_xy",
)

RUN_BRANCHES = (
    "Run",
    "TriggerRate",
    "RunDuration",
    "WorkingPoint",
)


def build_experimental_golden_schema() -> dict[str, str]:
    schema: dict[str, str] = {}

    for branch in TRACK_BRANCHES:
        schema[branch] = "float64"

    for suffix in PLANE_SUFFIXES:
        schema[f"TrackClusterMaxStripEnergy_{suffix}"] = "float64"
        schema[f"TrackClusterMinStripEnergy_{suffix}"] = "float64"

    for suffix in ("Z1", "Z2", "Z3", "Z4"):
        schema[f"TrackClusterEnergy_{suffix}"] = "float64"
        schema[f"TrackClusterPosition_{suffix}"] = "float64"
        schema[f"TrackClusterSize_{suffix}"] = "int64"

    for suffix in ("Y1", "Y2", "Y3", "Y4"):
        schema[f"TrackClusterEnergy_{suffix}"] = "float64"

    for suffix in PLANE_SUFFIXES:
        schema[f"TrackClusterExp_{suffix}"] = "float64"

    for suffix in ("Y1", "Y2", "Y3", "Y4"):
        schema[f"TrackClusterPosition_{suffix}"] = "float64"
        schema[f"TrackClusterSize_{suffix}"] = "int64"

    for suffix in PLANE_SUFFIXES:
        schema[f"ClusterMultriplicity_{suffix}"] = "int64"

    for suffix in PLANE_SUFFIXES:
        schema[f"NonTrackClusterEnergy_{suffix}"] = "var * float64"

    for suffix in PLANE_SUFFIXES:
        schema[f"NonTrackClusterSize_{suffix}"] = "var * int64"

    for suffix in PLANE_SUFFIXES:
        schema[f"NonTrackClusterPosition_{suffix}"] = "var * float64"

    schema["Run"] = "int64"
    schema["TriggerRate"] = "float64"
    schema["RunDuration"] = "float64"
    schema["WorkingPoint"] = "float64"

    return schema


def build_experimental_golden_record(
    *,
    internal_record: Mapping[str, Any],
    run: int,
    trigger_rate: float,
    run_duration: float,
    working_point: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for branch in TRACK_BRANCHES:
        output[branch] = internal_record[branch]

    for suffix in PLANE_SUFFIXES:
        output[f"TrackClusterMaxStripEnergy_{suffix}"] = internal_record[
            f"TrackClusterMaxStripEnergy_{suffix}"
        ]
        output[f"TrackClusterMinStripEnergy_{suffix}"] = internal_record[
            f"TrackClusterMinStripEnergy_{suffix}"
        ]

    for suffix in ("Z1", "Z2", "Z3", "Z4"):
        output[f"TrackClusterEnergy_{suffix}"] = internal_record[
            f"TrackClusterEnergy_{suffix}"
        ]
        output[f"TrackClusterPosition_{suffix}"] = internal_record[
            f"TrackClusterPosition_{suffix}"
        ]
        output[f"TrackClusterSize_{suffix}"] = internal_record[
            f"TrackClusterSize_{suffix}"
        ]

    for suffix in ("Y1", "Y2", "Y3", "Y4"):
        output[f"TrackClusterEnergy_{suffix}"] = internal_record[
            f"TrackClusterEnergy_{suffix}"
        ]

    for suffix in PLANE_SUFFIXES:
        output[f"TrackClusterExp_{suffix}"] = internal_record[
            f"TrackClusterTexp_{suffix}"
        ]

    for suffix in ("Y1", "Y2", "Y3", "Y4"):
        output[f"TrackClusterPosition_{suffix}"] = internal_record[
            f"TrackClusterPosition_{suffix}"
        ]
        output[f"TrackClusterSize_{suffix}"] = internal_record[
            f"TrackClusterSize_{suffix}"
        ]

    for suffix in PLANE_SUFFIXES:
        output[f"ClusterMultriplicity_{suffix}"] = internal_record[
            f"ClusterMultiplicity_{suffix}"
        ]

    for suffix in PLANE_SUFFIXES:
        output[f"NonTrackClusterEnergy_{suffix}"] = list(
            internal_record[f"NonTrackClusterEnergy_{suffix}"]
        )

    for suffix in PLANE_SUFFIXES:
        output[f"NonTrackClusterSize_{suffix}"] = list(
            internal_record[f"NonTrackClusterSize_{suffix}"]
        )

    for suffix in PLANE_SUFFIXES:
        output[f"NonTrackClusterPosition_{suffix}"] = list(
            internal_record[f"NonTrackClusterPosition_{suffix}"]
        )

    output["Run"] = int(run)
    output["TriggerRate"] = float(trigger_rate)
    output["RunDuration"] = float(run_duration)
    output["WorkingPoint"] = float(working_point)

    expected = tuple(build_experimental_golden_schema())
    if tuple(output) != expected:
        raise RuntimeError("Experimental golden branch order mismatch")

    return output
@dataclass(frozen=True)
class ExperimentalRunMetadata:
    run: int
    trigger_rate: float
    run_duration: float
    working_point: float


@dataclass(frozen=True)
class ExperimentalGoldenOutput:
    root_path: Path
    diagnostics: GoldenBatchDiagnostics


def run_experimental_golden_output(
    *,
    clusters: Iterable[ClusterRecord],
    reconstruct_event: Callable[[TrackingEvent], TrackingResult],
    run_metadata: ExperimentalRunMetadata,
    output_path: str | Path,
    tree_name: str = "Tracks4ptree",
) -> ExperimentalGoldenOutput:
    events = build_tracking_events(list(clusters))

    batch = process_golden_events(
        events,
        lambda event: build_processing_result(
            event,
            reconstruct_event(event),
        ),
    )

    internal_records = [
        {
            branch_name: batch.columns[branch_name][entry_index]
            for branch_name in batch.columns
        }
        for entry_index in range(batch.entry_count)
    ]

    experimental_records = [
        build_experimental_golden_record(
            internal_record=record,
            run=run_metadata.run,
            trigger_rate=run_metadata.trigger_rate,
            run_duration=run_metadata.run_duration,
            working_point=run_metadata.working_point,
        )
        for record in internal_records
    ]

    experimental_batch = build_golden_batch(
        selected_records=experimental_records,
        diagnostics=batch.diagnostics,
    )

    root_path = write_golden_root(
        batch=experimental_batch,
        output_path=output_path,
        tree_name=tree_name,
        empty_branch_types=build_experimental_golden_schema(),
    )

    return ExperimentalGoldenOutput(
        root_path=Path(root_path),
        diagnostics=batch.diagnostics,
    )
