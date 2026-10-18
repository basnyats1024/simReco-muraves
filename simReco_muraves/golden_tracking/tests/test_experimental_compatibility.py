from __future__ import annotations

from simReco_muraves.golden_tracking.experimental_compatibility import (
    PLANE_SUFFIXES,
    build_experimental_golden_record,
    build_experimental_golden_schema,
)


def internal_record():
    record = {
        "Phi": 180.0,
        "Theta": 22.5,
        "Scattering_xy": -1.0,
        "Scattering_xz": 1.0,
        "ChiSquare_xz": 3.0,
        "ChiSquare_xy": 4.0,
        "ChiSquare_3p_xz": 1.0,
        "ChiSquare_3p_xy": 2.0,
    }

    for suffix in PLANE_SUFFIXES:
        record[f"TrackClusterMaxStripEnergy_{suffix}"] = 10.0
        record[f"TrackClusterMinStripEnergy_{suffix}"] = 5.0
        record[f"TrackClusterEnergy_{suffix}"] = 15.0
        record[f"TrackClusterPosition_{suffix}"] = 2.0
        record[f"TrackClusterSize_{suffix}"] = 2
        record[f"TrackClusterTexp_{suffix}"] = 100.0
        record[f"ClusterMultiplicity_{suffix}"] = 3
        record[f"NonTrackClusterEnergy_{suffix}"] = [1.0, 2.0]
        record[f"NonTrackClusterSize_{suffix}"] = [1, 1]
        record[f"NonTrackClusterPosition_{suffix}"] = [3.0, 4.0]

    record["ProcessID"] = 1
    record["EventID"] = 10
    record["TrackClusterIndex_Z1"] = 0
    return record


def test_schema_contains_exact_experimental_branch_count():
    schema = build_experimental_golden_schema()
    assert len(schema) == 92


def test_schema_uses_legacy_experimental_names():
    schema = build_experimental_golden_schema()

    assert "TrackClusterExp_Z1" in schema
    assert "TrackClusterTexp_Z1" not in schema
    assert "ClusterMultriplicity_Z1" in schema
    assert "ClusterMultiplicity_Z1" not in schema


def test_builds_exact_experimental_projection():
    output = build_experimental_golden_record(
        internal_record=internal_record(),
        run=123,
        trigger_rate=4.5,
        run_duration=600.0,
        working_point=54.0,
    )

    assert tuple(output) == tuple(build_experimental_golden_schema())
    assert output["TrackClusterExp_Z1"] == 100.0
    assert output["ClusterMultriplicity_Z1"] == 3
    assert output["Run"] == 123
    assert output["TriggerRate"] == 4.5
    assert output["RunDuration"] == 600.0
    assert output["WorkingPoint"] == 54.0


def test_drops_internal_only_debug_branches():
    output = build_experimental_golden_record(
        internal_record=internal_record(),
        run=123,
        trigger_rate=4.5,
        run_duration=600.0,
        working_point=54.0,
    )

    assert "ProcessID" not in output
    assert "EventID" not in output
    assert "TrackClusterIndex_Z1" not in output


def test_vector_values_are_copied():
    source = internal_record()

    output = build_experimental_golden_record(
        internal_record=source,
        run=123,
        trigger_rate=4.5,
        run_duration=600.0,
        working_point=54.0,
    )

    output["NonTrackClusterEnergy_Z1"].append(99.0)

    assert source["NonTrackClusterEnergy_Z1"] == [1.0, 2.0]
