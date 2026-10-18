# simReco_muraves/io/provenance_tree.py

"""Store the provenance lineage inside the ROOT file it describes.

A sidecar file can be separated from its data; an in-file record
cannot. The lineage is written as a one-entry tree named Metadata,
holding four scalar branches for quick inspection plus the full chain
as JSON.

Everything here is best-effort: a provenance failure warns and returns
False rather than raising, so it can never abort a production run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import uproot

from ..provenance import (
    SCHEMA_VERSION,
    UNKNOWN,
    build_stage_record,
    chain_from_json,
    chain_to_json,
    extend_chain,
)


METADATA_TREE_NAME = "Metadata"

#: Fallback object name, used when the ROOT backend cannot write a
#: tree with string branches.
METADATA_OBJECT_NAME = "Metadata_json"

JSON_BRANCH = "provenance_json"


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def _chain_from_tree(root_file) -> list[dict[str, Any]] | None:
    if METADATA_TREE_NAME not in root_file:
        return None

    try:
        tree = root_file[METADATA_TREE_NAME]

        if JSON_BRANCH not in tree:
            return None

        values = tree[JSON_BRANCH].array(library="np")

        if len(values) == 0:
            return []

        text = values[0]

        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        return chain_from_json(str(text))
    except Exception:
        return None


def _chain_from_object(root_file) -> list[dict[str, Any]] | None:
    if METADATA_OBJECT_NAME not in root_file:
        return None

    try:
        stored = root_file[METADATA_OBJECT_NAME]

        for accessor in (
            lambda item: item.member("fString"),
            lambda item: item.fString,
            str,
        ):
            try:
                text = accessor(stored)
            except Exception:
                continue

            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")

            parsed = chain_from_json(str(text))

            if parsed:
                return parsed

        return None
    except Exception:
        return None


def read_chain(path: str | Path) -> list[dict[str, Any]]:
    """Return the lineage stored in a ROOT file, or [] if there is none.

    An absent record is normal: Geant4 input files have none, and so do
    outputs produced before provenance recording existed.
    """
    file_path = Path(path)

    if not file_path.is_file():
        return []

    try:
        with uproot.open(file_path) as root_file:
            from_tree = _chain_from_tree(root_file)

            if from_tree is not None:
                return from_tree

            from_object = _chain_from_object(root_file)

            if from_object is not None:
                return from_object
    except Exception:
        return []

    return []


def read_summary(path: str | Path) -> dict[str, Any]:
    """Scalar summary of the last stage that touched a file."""
    chain = read_chain(path)

    if not chain:
        return {
            "stages": 0,
            "stage": UNKNOWN,
            "timestamp": UNKNOWN,
            "code_version": UNKNOWN,
            "schema_version": UNKNOWN,
        }

    last = chain[-1]

    return {
        "stages": len(chain),
        "stage": last.get("stage", UNKNOWN),
        "timestamp": last.get("timestamp", UNKNOWN),
        "code_version": last.get("code_version", UNKNOWN),
        "schema_version": last.get("schema_version", UNKNOWN),
    }


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _tree_payload(chain: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import awkward as ak
    import numpy as np

    last = chain[-1] if chain else {}

    return {
        "schema_version": np.array(
            [int(last.get("schema_version", SCHEMA_VERSION))],
            dtype=np.int32,
        ),
        "stage_count": np.array([len(chain)], dtype=np.int32),
        "git_commit": ak.Array(
            [str(last.get("code_version", UNKNOWN))]
        ),
        "stage": ak.Array([str(last.get("stage", UNKNOWN))]),
        "timestamp": ak.Array(
            [str(last.get("timestamp", UNKNOWN))]
        ),
        JSON_BRANCH: ak.Array([chain_to_json(chain)]),
    }


def write_chain(
    path: str | Path,
    chain: Sequence[Mapping[str, Any]],
) -> bool:
    """Write a lineage into an existing ROOT file.

    Tries a one-entry tree first and falls back to a stored string if
    the backend cannot write string branches. Returns True on success.
    """
    file_path = Path(path)

    try:
        with uproot.update(file_path) as root_file:
            root_file[METADATA_TREE_NAME] = _tree_payload(chain)

        return True
    except Exception as tree_error:
        try:
            with uproot.update(file_path) as root_file:
                root_file[METADATA_OBJECT_NAME] = chain_to_json(chain)

            return True
        except Exception as object_error:
            warnings.warn(
                "Could not write provenance into "
                f"{file_path.name}: tree write failed with "
                f"{type(tree_error).__name__}: {tree_error}; "
                "string fallback failed with "
                f"{type(object_error).__name__}: {object_error}",
                RuntimeWarning,
                stacklevel=2,
            )

            return False


def attach_provenance(
    output_path: str | Path,
    *,
    stage: str,
    input_path: str | Path | None = None,
    parameters: Mapping[str, Any] | None = None,
    quiet: bool = False,
) -> bool:
    """Append this stage to the lineage carried by its input.

    Reads whatever provenance the input file holds, appends a record
    for this stage, and writes the result into the output file.
    Returns True on success; never raises.
    """
    try:
        upstream = read_chain(input_path) if input_path else []

        record = build_stage_record(
            stage=stage,
            input_file=input_path,
            output_file=output_path,
            parameters=parameters,
        )

        chain = extend_chain(upstream, record)

        written = write_chain(output_path, chain)

        if written and not quiet:
            print(
                f"  provenance      : {len(chain)} stage(s) recorded"
            )

        return written
    except Exception as error:
        warnings.warn(
            "Provenance could not be recorded for "
            f"{Path(output_path).name}: "
            f"{type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

        return False


def attach_provenance_to_many(
    output_paths: Sequence[str | Path],
    *,
    stage: str,
    input_path: str | Path | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> int:
    """Attach the same stage record to several output files.

    Used by clusterization, whose output is split into chunks. Returns
    the number of files successfully annotated.
    """
    written = 0

    for output_path in output_paths:
        if attach_provenance(
            output_path,
            stage=stage,
            input_path=input_path,
            parameters=parameters,
            quiet=True,
        ):
            written += 1

    total = len(output_paths)

    if written == total:
        print(f"  provenance      : recorded in {written} file(s)")
    else:
        print(
            f"  provenance      : recorded in {written} of "
            f"{total} file(s)"
        )

    return written
