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
def base_petri_matrix():
    """A simple, deterministic Petri net for testing."""
    return np.array([[1, 0, 0, 1, 1], [0, 1, 1, 0, 0]])


def test_generate_candidate_matrices(base_petri_matrix):
    """Test the generation of candidate matrices."""
    config = {
        "enable_add_edge": True,
        "enable_delete_edge": True,
        "enable_add_place": True,
        "enable_add_token": True,
        "enable_delete_token": True,
    }
    candidates = _generate_candidate_matrices(base_petri_matrix, config)
    assert len(candidates) > 0
    # Check that each candidate is a valid Petri net matrix
    for c in candidates:
        assert isinstance(c, np.ndarray)
        assert c.shape[1] == base_petri_matrix.shape[1] or c.shape[1] == 0


@patch("DataGenerate.SPN.generate_stochastic_net_task_with_rates")
def test_generate_rate_variations(mock_spn_task, base_petri_matrix):
    """Test the generation of rate variations."""
    mock_spn_task.return_value = (None, None, np.array([1.0]), True)
    base_variation = {
        "petri_net": base_petri_matrix,
        "arr_vlist": [np.array([1, 0])],
        "arr_edge": np.array([[0, 1]]),
        "arr_tranidx": np.array([0]),
    }
    variations = _generate_rate_variations(base_variation, 5)
    assert len(variations) == 5
    assert mock_spn_task.call_count == 5


@patch("DataGenerate.SPN.filter_spn")
@patch("DataGenerate.DataTransformation._generate_rate_variations")
def test_generate_petri_net_variations(mock_rate_variations, mock_filter_spn, base_petri_matrix):
    """Test the generation of Petri net variations."""
    mock_filter_spn.return_value = ({"petri_net": base_petri_matrix}, True)
    mock_rate_variations.return_value = [{"petri_net": base_petri_matrix}]
    # Config must enable some structural change for rate variations to be generated
    config = {"enable_add_edge": True, "enable_rate_variations": True}
    variations = generate_petri_net_variations(base_petri_matrix, config)
    assert len(variations) > 0


@patch("DataGenerate.SPN.get_spn_info")
def test_generate_lambda_variations(mock_get_spn_info, base_petri_matrix):
    """Test the generation of lambda variations, preventing infinite loops."""
    # Mock the function that is actually called inside the loop
    mock_get_spn_info.return_value = ({}, True)

    # Provide a valid list of vertices to prevent an immediate fail state
    # that would cause an infinite loop.
    petri_dict = {
        "petri_net": base_petri_matrix,
        "arr_vlist": [np.array([1, 0])],  # Must be non-empty
        "arr_edge": np.array([[0, 1]]),
        "arr_tranidx": np.array([0]),
    }

    variations = generate_lambda_variations(petri_dict, 5)

    # Assert that the function produced the correct number of variations
    # and that our mock was called as expected.
    assert len(variations) == 5
    assert mock_get_spn_info.call_count == 5
