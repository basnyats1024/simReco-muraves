from __future__ import annotations

from ..nested_access import nested_value_strict
from .event_cluster_output import (
    PLANE_NAMES,
    PlaneClusterInput,
)
from .golden_pipeline import (
    GoldenProcessingResult,
    consistency_error_result,
)
from .golden_record_builder import build_golden_record
from ..models.tracking_event import TrackingEvent
from ..models.tracking_results import TrackingResult


def _plane_input(clusters) -> PlaneClusterInput:
    return PlaneClusterInput(
        energies=[cluster.cluster_energy for cluster in clusters],
        positions=[cluster.cluster_position for cluster in clusters],
        sizes=[cluster.cluster_size for cluster in clusters],
        texp=[cluster.cluster_time for cluster in clusters],
        strip_energies=[cluster.strip_energies for cluster in clusters],
    )


def _selected_nested_value(
    values,
    parent_index: int,
    inner_index: int,
    field_name: str,
) -> float:
    """Deprecated: use simReco_muraves.nested_access.nested_value_strict.

    Kept so that existing callers and tests continue to work.
    """
    return nested_value_strict(
        values,
        parent_index,
        inner_index,
        field_name,
    )


def build_event_plane_inputs(
    event: TrackingEvent,
) -> dict[str, PlaneClusterInput]:
    """Cluster payload of a TrackingEvent, keyed by plane name."""
    return {
        plane_name: _plane_input(getattr(event, plane_name))
        for plane_name in PLANE_NAMES
    }


def build_processing_result(
    event: TrackingEvent,
    result: TrackingResult,
) -> GoldenProcessingResult:
    """Golden projection for the clustered route.

    Checks event identity, then hands the selected state to the shared
    record builder used by both routes.
    """
    if event.process_id != result.process_id or event.event != result.event:
        return consistency_error_result("event_identity_mismatch")

    return build_golden_record(
        planes=build_event_plane_inputs(event),
        result=result,
    )
