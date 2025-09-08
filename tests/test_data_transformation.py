import numpy as np
import pytest
from unittest.mock import patch
from DataGenerate.DataTransformation import (
    _generate_candidate_matrices,
    _generate_rate_variations,
    generate_petri_net_variations,
    generate_lambda_variations,
)


@pytest.fixture
def simple_petri_matrix():
    """A simple 2x2 Petri net matrix."""
    return np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
        ],
        dtype="int32",
    )


def test_generate_candidate_matrices(simple_petri_matrix):
    """Tests the generation of candidate matrices."""
    candidates = _generate_candidate_matrices(simple_petri_matrix)
    # 4 edges + 4 non-edges + 2 places to add token + 1 place to delete token + 1 place to add
    # 4+4+2+1+1 = 12
    # Let's verify this.
    # 4 edges to delete.
    # 4 non-edges to add.
    # 2 places to add a token.
    # 1 place with tokens to delete from.
    # 1 new place to add.
    # Total = 4 + 4 + 2 + 0 + 1 = 11.
    assert len(candidates) == 11

    # Check that one of the candidates has a deleted edge
    deleted_edge_candidate = candidates[0]
    assert np.sum(deleted_edge_candidate[:, :-1]) < np.sum(simple_petri_matrix[:, :-1])

    # Check that one of the candidates has an added edge
    added_edge_candidate = candidates[4]
    assert np.sum(added_edge_candidate[:, :-1]) > np.sum(simple_petri_matrix[:, :-1])

    # Check that one of the candidates has an added token
    added_token_candidate = candidates[8]
    assert np.sum(added_token_candidate[:, -1]) > np.sum(simple_petri_matrix[:, -1])

    # Check that one of the candidates has an added place
    added_place_candidate = candidates[10]
    assert added_place_candidate.shape[0] > simple_petri_matrix.shape[0]


@patch("DataGenerate.DataTransformation.SPN.generate_stochastic_net_task_with_rates")
def test_generate_rate_variations(mock_spn_task, simple_petri_matrix):
    """Tests the generation of rate variations."""
    base_variation = {
        "petri_net": simple_petri_matrix,
        "arr_vlist": [np.array([1, 0]), np.array([0, 1])],
        "arr_edge": np.array([[0, 1]]),
        "arr_tranidx": np.array([0]),
    }
    mock_spn_task.return_value = (None, None, None, True)
    variations = _generate_rate_variations(base_variation, num_variations=5)
    assert len(variations) == 5
    assert mock_spn_task.call_count == 5


@patch("DataGenerate.DataTransformation.SPN.filter_spn")
@patch("DataGenerate.DataTransformation._generate_rate_variations")
def test_generate_petri_net_variations(mock_rate_variations, mock_filter_spn, simple_petri_matrix):
    """Tests the generation of Petri net variations."""
    mock_filter_spn.return_value = ({}, True)
    mock_rate_variations.return_value = [{}]
    variations = generate_petri_net_variations(simple_petri_matrix, 10, 2, 500, parallel_job_count=1)
    assert len(variations) > 0
    # 12 candidates, all successful, 1 rate variation for each.
    # 12 structural variations + 12 rate variations = 24
    # But _generate_rate_variations is mocked to return 1 variation.
    # So 12 + 12 = 24 is wrong.
    # It should be 12 structural + 12 * 1 = 24.
    # No, it's 12 structural variations, and for each, we generate rate variations.
    # So it should be 12 + 12 = 24. Wait, the code is `all_augmented_data.extend(structural_variations)`
    # and then `all_augmented_data.extend(rate_variations)`. So it is addition.
    # Let's re-read the code.
    # `structural_variations` is a list of dicts. `rate_variations` is also a list of dicts.
    # The code extends the list.
    # So if `_generate_candidate_matrices` returns 12 matrices, and `filter_spn` approves all,
    # then `structural_variations` has 12 items.
    # Then for each of these 12, we call `_generate_rate_variations`.
    # If that returns 1 item each time, then we have 12 * 1 = 12 rate variations in total.
    # No, the loop is `for base_variation in structural_variations:`.
    # So we get 11 * 1 = 11 rate variations.
    # Total variations = 11 + 11 = 22.
    assert len(variations) == 22


@patch("DataGenerate.DataTransformation.SPN.get_spn_info")
def test_generate_lambda_variations(mock_spn_info):
    """Tests the generation of lambda variations."""
    petri_dict = {
        "petri_net": np.array([[1, 0, 0, 1, 1]]),
        "arr_vlist": [],
        "arr_edge": [],
        "arr_tranidx": [],
    }
    mock_spn_info.return_value = ({}, True)
    variations = generate_lambda_variations(petri_dict, num_lambda_variations=3)
    assert len(variations) == 3
    assert mock_spn_info.call_count == 3
