from __future__ import annotations

import awkward as ak
import numpy as np

from pathlib import Path
from typing import Mapping

from .batch_output import (
    GoldenBatch,
    validate_column_lengths,
)


DEFAULT_TREE_NAME = "Tracks4ptree"


def _import_root_dependencies():
    try:
        import awkward as ak
        import numpy as np
        import uproot
    except ImportError as error:
        raise RuntimeError(
            "ROOT output requires uproot, awkward, and numpy"
        ) from error

    return uproot, ak, np


def _convert_column(values, *, ak, np):
    if not values:
        return np.asarray([])

    first_value = values[0]

    if isinstance(first_value, (list, tuple)):
        return ak.Array(values)

    if isinstance(first_value, bool):
        return np.asarray(values, dtype=np.bool_)

    if isinstance(first_value, int):
        return np.asarray(values, dtype=np.int64)

    if isinstance(first_value, float):
        return np.asarray(values, dtype=np.float64)

    raise TypeError(
        "Unsupported ROOT branch value type: "
        f"{type(first_value).__name__}"
    )


def _convert_columns(columns, *, ak, np):
    return {
        branch_name: _convert_column(
            values,
            ak=ak,
            np=np,
        )
        for branch_name, values in columns.items()
    }


def _typed_jagged_array(values, dtype):
    offsets = [0]
    flattened = []

    for value in values:
        flattened.extend(value)
        offsets.append(len(flattened))

    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(offsets, dtype=np.int64)),
            ak.contents.NumpyArray(
                np.asarray(flattened, dtype=dtype)
            ),
        )
    )



def write_golden_root(
    *,
    batch: GoldenBatch,
    output_path: str | Path,
    tree_name: str = DEFAULT_TREE_NAME,
    empty_branch_types: Mapping[str, str] | None = None,
) -> Path:
    uproot, ak, np = _import_root_dependencies()

    if not isinstance(tree_name, str) or not tree_name:
        raise ValueError(
            "tree_name must be a non-empty string"
        )

    entry_count = validate_column_lengths(batch.columns)

    if entry_count != batch.entry_count:
        raise ValueError(
            "GoldenBatch entry count is inconsistent"
        )

    if entry_count != batch.diagnostics.selected_events:
        raise ValueError(
            "ROOT entry count does not match selected-event count"
        )

    path = Path(output_path)

    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with uproot.recreate(path) as root_file:
        if entry_count > 0:
            if empty_branch_types:
                tree = root_file.mktree(
                    tree_name,
                    dict(empty_branch_types),
                )
                tree.extend(
                    _convert_columns(
                        batch.columns,
                        ak=ak,
                        np=np,
                    )
                )
            else:
                root_file[tree_name] = _convert_columns(
                    batch.columns,
                    ak=ak,
                    np=np,
                )
        else:
            if not empty_branch_types:
                raise ValueError(
                    "empty_branch_types are required for "
                    "a zero-entry ROOT output"
                )

            root_file.mktree(
                tree_name,
                dict(empty_branch_types),
            )

    return path
