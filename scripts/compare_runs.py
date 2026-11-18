#!/usr/bin/env python3
"""Compare two chain outputs branch by branch.

    python compare_runs.py RUN_A RUN_B

Each RUN is an output directory written by run_chain.sh, containing
clustered/, tracked/ and golden/. For every chunk in each stage, the
physics tree is compared entry by entry and branch by branch, exactly,
with NaN treated as equal to NaN.

Comparing the files byte for byte would not work: ROOT embeds
timestamps, and each file's Metadata tree records when and where it
was produced. The provenance trees are therefore deliberately left
out. Everything else must match.

Exit status is 0 only if every stage, file, branch and entry agrees.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import awkward as ak
import uproot


STAGES = (
    ("clustered", "ClusteredData", "*_clustered_part*.root"),
    ("tracked", "TrackedData", "*_tracked_part*.root"),
    ("golden", "Tracks4ptree", "*_golden_part*.root"),
)

MAX_REPORTED_FILES = 10


def values_equal(left, right) -> bool:
    """Exact, nested, NaN-aware equality for values from tolist()."""
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(values_equal(a, b) for a, b in zip(left, right))
        )

    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True

    return left == right


def first_difference(left, right):
    """Locate the first differing entry, for the report only."""
    left_values = left.tolist()
    right_values = right.tolist()

    for index, (a, b) in enumerate(zip(left_values, right_values)):
        if not values_equal(a, b):
            return index, a, b

    return None


def compare_file(path_a: Path, path_b: Path, tree_name: str) -> list[str]:
    problems: list[str] = []

    with uproot.open(path_a) as file_a, uproot.open(path_b) as file_b:
        tree_a = file_a[tree_name]
        tree_b = file_b[tree_name]

        if tree_a.num_entries != tree_b.num_entries:
            return [
                f"entry count {tree_a.num_entries} vs {tree_b.num_entries}"
            ]

        branches_a = set(tree_a.keys())
        branches_b = set(tree_b.keys())

        if branches_a != branches_b:
            problems.append(
                "branch sets differ: "
                f"only in A {sorted(branches_a - branches_b)[:5]}, "
                f"only in B {sorted(branches_b - branches_a)[:5]}"
            )

        for branch in sorted(branches_a & branches_b):
            array_a = tree_a[branch].array(library="ak")
            array_b = tree_b[branch].array(library="ak")

            if ak.array_equal(array_a, array_b, equal_nan=True):
                continue

            difference = first_difference(array_a, array_b)

            if difference is None:
                problems.append(f"{branch}: differs (type or structure)")
            else:
                index, a, b = difference
                problems.append(
                    f"{branch}: first difference at entry {index}: "
                    f"{a!r} vs {b!r}"
                )

    return problems


def count_entries(paths: list[Path], tree_name: str) -> int:
    total = 0

    for path in paths:
        with uproot.open(path) as root_file:
            total += root_file[tree_name].num_entries

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two run_chain.sh outputs branch by branch.",
    )
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    arguments = parser.parse_args()

    for run in (arguments.run_a, arguments.run_b):
        if not run.is_dir():
            raise SystemExit(f"not a directory: {run}")

    print()
    print("==================================================")
    print("Reproducibility comparison")
    print("==================================================")
    print(f"A : {arguments.run_a.resolve()}")
    print(f"B : {arguments.run_b.resolve()}")

    failed_files: list[str] = []
    stage_problems: list[str] = []

    for stage, tree_name, pattern in STAGES:
        files_a = sorted((arguments.run_a / stage).glob(pattern))
        files_b = sorted((arguments.run_b / stage).glob(pattern))
        names_a = [path.name for path in files_a]
        names_b = [path.name for path in files_b]

        print()
        print(f"{stage} ({tree_name})")
        print("--------------------------------------------------")
        print(f"  files      : {len(files_a)} vs {len(files_b)}")

        if not files_a and not files_b:
            stage_problems.append(f"{stage}: no files in either run")
            print("  no files in either run")
            continue

        if names_a != names_b:
            only_a = sorted(set(names_a) - set(names_b))
            only_b = sorted(set(names_b) - set(names_a))
            stage_problems.append(f"{stage}: file lists differ")
            print(f"  only in A  : {only_a[:5]}")
            print(f"  only in B  : {only_b[:5]}")

        common = sorted(set(names_a) & set(names_b))
        identical = 0

        for position, name in enumerate(common, start=1):
            print(
                f"\r  comparing  : {position}/{len(common)}",
                end="",
                flush=True,
            )

            problems = compare_file(
                arguments.run_a / stage / name,
                arguments.run_b / stage / name,
                tree_name,
            )

            if not problems:
                identical += 1
                continue

            failed_files.append(f"{stage}/{name}")

            if len(failed_files) <= MAX_REPORTED_FILES:
                print(f"\n  DIFFERS    : {name}")
                for problem in problems[:5]:
                    print(f"      {problem}")

        print(f"\r  identical  : {identical}/{len(common)} files")

        entries_a = count_entries(files_a, tree_name)
        entries_b = count_entries(files_b, tree_name)
        print(f"  entries    : {entries_a:,} vs {entries_b:,}")

    print()
    print("==================================================")

    if failed_files or stage_problems:
        print("NOT reproducible")

        for problem in stage_problems:
            print(f"  {problem}")

        if failed_files:
            print(f"  {len(failed_files)} file(s) differ")

        print("==================================================")
        sys.exit(1)

    print("Fully reproducible: every stage, file, branch and entry agrees.")
    print("==================================================")


if __name__ == "__main__":
    main()
