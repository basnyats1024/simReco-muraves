# simReco_muraves/provenance.py

"""Record of what produced a reconstruction output file.

Every stage of the chain appends one entry to a lineage carried
forward from its input, so a golden file knows not only the tracking
parameters that made it but the digitization constant that made the
clusters it descends from.

This module builds and chains those records. It does no I/O on ROOT
files; see simReco_muraves.io.provenance_tree for that.

Collection is best-effort by design. A provenance failure must never
abort a production run, so every lookup here degrades to "unknown"
rather than raising.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


#: Layout revision of the golden output tree.
#:
#: Bump when branches are added, removed, renamed, or change meaning.
#: Do not bump for value changes: a different digitization constant
#: produces different numbers in the same schema, and the parameter
#: record below is what distinguishes those files.
#:
#:   1  first versioned layout
#:   2  adds BestTrack4pIsTexpNull_xz / _xy
SCHEMA_VERSION = 2

UNKNOWN = "unknown"

#: Libraries whose behaviour the output can depend on.
TRACKED_LIBRARIES = ("uproot", "awkward", "numpy")

DISTRIBUTION_NAMES = ("simreco-muraves", "simReco_muraves")

_GIT_TIMEOUT_SECONDS = 5


# ----------------------------------------------------------------------
# Environment lookups
# ----------------------------------------------------------------------

def _run_git(arguments: Sequence[str]) -> str | None:
    repository_root = Path(__file__).resolve().parent.parent

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def git_state() -> tuple[str, bool]:
    """Return (commit hash, has uncommitted changes).

    Returns ("unknown", False) outside a git checkout, which is the
    normal case for a container or an unpacked tarball.
    """
    commit = _run_git(["rev-parse", "HEAD"])

    if commit is None:
        return UNKNOWN, False

    status = _run_git(["status", "--porcelain"])
    dirty = bool(status)

    return commit, dirty


def package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return UNKNOWN

    for name in DISTRIBUTION_NAMES:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            return UNKNOWN

    return UNKNOWN


def library_versions() -> dict[str, str]:
    """Versions of the libraries the output format depends on."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}

    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = UNKNOWN
        except Exception:
            versions[name] = UNKNOWN

    return versions


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# Stage records
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class StageRecord:
    stage: str
    timestamp: str
    input_file: str
    output_file: str
    parameters: dict[str, Any] = field(default_factory=dict)
    code_version: str = UNKNOWN
    code_dirty: bool = False
    package_version: str = UNKNOWN
    libraries: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "parameters": dict(self.parameters),
            "code_version": self.code_version,
            "code_dirty": self.code_dirty,
            "package_version": self.package_version,
            "libraries": dict(self.libraries),
            "schema_version": self.schema_version,
        }


def _jsonable(value: Any) -> Any:
    """Coerce a parameter value into something JSON can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]

    return str(value)


def build_stage_record(
    *,
    stage: str,
    input_file: str | Path | None,
    output_file: str | Path,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one stage of the chain."""
    commit, dirty = git_state()

    record = StageRecord(
        stage=stage,
        timestamp=utc_timestamp(),
        input_file=str(input_file) if input_file is not None else "",
        output_file=str(output_file),
        parameters={
            str(key): _jsonable(value)
            for key, value in (parameters or {}).items()
        },
        code_version=commit,
        code_dirty=dirty,
        package_version=package_version(),
        libraries=library_versions(),
    )

    return record.to_dict()


def extend_chain(
    upstream: Sequence[Mapping[str, Any]] | None,
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Append a stage record to the lineage inherited from the input."""
    chain: list[dict[str, Any]] = [
        dict(entry) for entry in (upstream or [])
    ]

    chain.append(dict(record))

    return chain


def chain_to_json(chain: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(chain), sort_keys=True)


def chain_from_json(text: str) -> list[dict[str, Any]]:
    """Parse a stored lineage, tolerating anything unreadable."""
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    return [entry for entry in parsed if isinstance(entry, dict)]


# ----------------------------------------------------------------------
# Presentation
# ----------------------------------------------------------------------

def summarize_chain(chain: Sequence[Mapping[str, Any]]) -> str:
    """Human-readable lineage, for muraves-sim --provenance."""
    if not chain:
        return (
            "No provenance recorded.\n\n"
            "The file predates provenance recording, or metadata "
            "could not be written when it was produced."
        )

    lines: list[str] = []

    for depth, entry in enumerate(chain, start=1):
        stage = entry.get("stage", UNKNOWN)
        timestamp = entry.get("timestamp", UNKNOWN)

        lines.append(f"[{depth}] {stage}   {timestamp}")

        input_file = entry.get("input_file") or "(none)"
        lines.append(f"      input          : {input_file}")
        lines.append(
            f"      output         : {entry.get('output_file', UNKNOWN)}"
        )

        commit = entry.get("code_version", UNKNOWN)

        if entry.get("code_dirty"):
            commit = f"{commit} (uncommitted changes)"

        lines.append(f"      code           : {commit}")
        lines.append(
            "      package        : "
            f"{entry.get('package_version', UNKNOWN)}"
        )
        lines.append(
            "      schema version : "
            f"{entry.get('schema_version', UNKNOWN)}"
        )

        libraries = entry.get("libraries") or {}

        if libraries:
            rendered = ", ".join(
                f"{name} {value}"
                for name, value in sorted(libraries.items())
            )
            lines.append(f"      libraries      : {rendered}")

        parameters = entry.get("parameters") or {}

        if parameters:
            lines.append("      parameters     :")

            for key in sorted(parameters):
                lines.append(f"        {key} = {parameters[key]}")

        lines.append("")

    return "\n".join(lines).rstrip()
