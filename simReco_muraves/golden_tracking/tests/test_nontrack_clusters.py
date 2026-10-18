from simReco_muraves.golden_tracking.event_cluster_output import extract_nontrack_clusters


def test_excludes_selected_cluster_by_index():
    result = extract_nontrack_clusters(
        energies=[10.0, 20.0, 30.0],
        positions=[1.0, 2.0, 3.0],
        sizes=[1, 2, 3],
        selected_index=1,
    )

    assert result.energies == [10.0, 30.0]
    assert result.positions == [1.0, 3.0]
    assert result.sizes == [1, 3]


def test_selected_first_cluster():
    result = extract_nontrack_clusters(
        energies=[10.0, 20.0, 30.0],
        positions=[1.0, 2.0, 3.0],
        sizes=[1, 2, 3],
        selected_index=0,
    )

    assert result.energies == [20.0, 30.0]
    assert result.positions == [2.0, 3.0]
    assert result.sizes == [2, 3]


def test_selected_last_cluster():
    result = extract_nontrack_clusters(
        energies=[10.0, 20.0, 30.0],
        positions=[1.0, 2.0, 3.0],
        sizes=[1, 2, 3],
        selected_index=2,
    )

    assert result.energies == [10.0, 20.0]
    assert result.positions == [1.0, 2.0]
    assert result.sizes == [1, 2]


def test_single_selected_cluster_returns_empty_collections():
    result = extract_nontrack_clusters(
        energies=[10.0],
        positions=[1.0],
        sizes=[2],
        selected_index=0,
    )

    assert result.energies == []
    assert result.positions == []
    assert result.sizes == []


def test_empty_cluster_collections():
    result = extract_nontrack_clusters(
        energies=[],
        positions=[],
        sizes=[],
        selected_index=-1,
    )

    assert result.energies == []
    assert result.positions == []
    assert result.sizes == []


def test_invalid_negative_selected_index_keeps_all_clusters():
    result = extract_nontrack_clusters(
        energies=[10.0, 20.0],
        positions=[1.0, 2.0],
        sizes=[1, 2],
        selected_index=-1,
    )

    assert result.energies == [10.0, 20.0]
    assert result.positions == [1.0, 2.0]
    assert result.sizes == [1, 2]


def test_out_of_range_selected_index_keeps_all_clusters():
    result = extract_nontrack_clusters(
        energies=[10.0, 20.0],
        positions=[1.0, 2.0],
        sizes=[1, 2],
        selected_index=2,
    )

    assert result.energies == [10.0, 20.0]
    assert result.positions == [1.0, 2.0]
    assert result.sizes == [1, 2]


def test_exclusion_is_index_based_not_value_based():
    result = extract_nontrack_clusters(
        energies=[10.0, 10.0, 20.0],
        positions=[1.0, 1.0, 2.0],
        sizes=[1, 1, 2],
        selected_index=0,
    )

    assert result.energies == [10.0, 20.0]
    assert result.positions == [1.0, 2.0]
    assert result.sizes == [1, 2]


def test_preserves_original_cluster_order():
    result = extract_nontrack_clusters(
        energies=[30.0, 10.0, 20.0, 40.0],
        positions=[3.0, 1.0, 2.0, 4.0],
        sizes=[3, 1, 2, 4],
        selected_index=2,
    )

    assert result.energies == [30.0, 10.0, 40.0]
    assert result.positions == [3.0, 1.0, 4.0]
    assert result.sizes == [3, 1, 4]


def test_input_collections_are_not_modified():
    energies = [10.0, 20.0, 30.0]
    positions = [1.0, 2.0, 3.0]
    sizes = [1, 2, 3]

    extract_nontrack_clusters(
        energies=energies,
        positions=positions,
        sizes=sizes,
        selected_index=1,
    )

    assert energies == [10.0, 20.0, 30.0]
    assert positions == [1.0, 2.0, 3.0]
    assert sizes == [1, 2, 3]


def test_rejects_mismatched_collection_lengths():
    try:
        extract_nontrack_clusters(
            energies=[10.0, 20.0],
            positions=[1.0],
            sizes=[1, 2],
            selected_index=0,
        )
    except ValueError as error:
        assert "same length" in str(error).lower()
    else:
        raise AssertionError("Expected ValueError for mismatched collection lengths")


def test_result_collections_are_independent_lists():
    energies = [10.0, 20.0]
    positions = [1.0, 2.0]
    sizes = [1, 2]

    result = extract_nontrack_clusters(
        energies=energies,
        positions=positions,
        sizes=sizes,
        selected_index=-1,
    )

    assert result.energies is not energies
    assert result.positions is not positions
    assert result.sizes is not sizes
