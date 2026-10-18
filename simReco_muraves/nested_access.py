# simReco_muraves/nested_access.py

"""Bounds-checked access to doubly-nested four-plane quantities.

Four-plane tracking quantities (chi-squares, scattering angles, slopes,
cluster indices) are stored as nested sequences indexed by

    values[parent_3p_row][inner_4p_extension]

Both the Phase II reconstruction adapter and the Phase III golden
projection need exactly the same bounds checks on those two indices.
Having each side implement its own check is what allowed the two golden
paths to diverge, so the checks live here and are imported by both.

This module deliberately sits at package root rather than under
``io/`` or ``golden_tracking/``: it is pure index arithmetic with no
ROOT and no Phase III dependency, so ``tracking/`` can import it without
depending on the golden-track layer.
"""

from __future__ import annotations

import operator
from typing import Any, Sequence


#: Value used throughout the chain to mean "this quantity is unavailable".
MISSING_VALUE = -1.0


def _as_index(value: Any) -> int | None:
    """Return ``value`` as an integer index, or None if it is not one.

    Accepts Python ints and numpy integers; rejects bools and floats.
    """
    if isinstance(value, bool):
        return None

    try:
        return operator.index(value)
    except TypeError:
        return None


def nested_value(
    values: Sequence[Sequence[Any]],
    parent_index: Any,
    inner_index: Any,
) -> Any | None:
    """Return ``values[parent_index][inner_index]``.

    Returns None when either index is not an integer or falls outside
    its sequence, rather than raising.
    """
    parent = _as_index(parent_index)
    inner = _as_index(inner_index)

    if parent is None or inner is None:
        return None

    if not 0 <= parent < len(values):
        return None

    row = values[parent]

    if not 0 <= inner < len(row):
        return None

    return row[inner]


def nested_value_or_missing(
    values: Sequence[Sequence[Any]],
    parent_index: Any,
    inner_index: Any,
    *,
    missing: float = MISSING_VALUE,
) -> float:
    """Sentinel form: unreachable indices yield ``missing``.

    Used for the selected four-plane chi-squares, where an unreachable
    index is a normal physics outcome (no promoted four-plane track) and
    is signalled downstream by the sentinel value.
    """
    value = nested_value(values, parent_index, inner_index)

    if value is None:
        return float(missing)

    return float(value)


def nested_value_strict(
    values: Sequence[Sequence[Any]],
    parent_index: Any,
    inner_index: Any,
    field_name: str,
) -> float:
    """Raising form: unreachable indices are a consistency error.

    Used once the event has already passed the golden gate, where an
    unreachable index means the persisted selected state is internally
    inconsistent rather than simply absent.
    """
    value = nested_value(values, parent_index, inner_index)

    if value is None:
        raise IndexError(
            f"Invalid nested index for {field_name}: "
            f"parent={parent_index!r}, inner={inner_index!r}"
        )

    return float(value)
