import numpy as np
import pytest
from spn_datasets.generator.PetriGenerate import (
    generate_random_petri_net,
    prune_petri_net,
    delete_excess_edges,
    add_missing_connections,
    add_tokens_randomly,
)


def test_generate_random_petri_net():
    """Test the generation of a random Petri net."""
    num_places = 5
    num_transitions = 3
    petri_matrix = generate_random_petri_net(num_places, num_transitions)
    assert petri_matrix.shape == (num_places, 2 * num_transitions + 1)
    assert np.sum(petri_matrix[:, -1]) > 0  # Should have an initial marking


def test_prune_petri_net():
    """Test the pruning of a Petri net."""
    num_places = 10
    num_transitions = 5
    petri_matrix = generate_random_petri_net(num_places, num_transitions)
    pruned_matrix = prune_petri_net(petri_matrix)
    assert pruned_matrix.shape == petri_matrix.shape


def test_delete_excess_edges():
    """Test the deletion of excess edges."""
    petri_matrix = np.ones((5, 11), dtype=int)
    num_transitions = 5
    modified_matrix = delete_excess_edges(petri_matrix, num_transitions)
    # Each place and transition should have at most 2 edges
    for i in range(5):
        assert np.sum(modified_matrix[i, :-1]) <= 2
    for i in range(10):
        assert np.sum(modified_matrix[:, i]) <= 2


def test_add_missing_connections():
    """Test adding missing connections."""
    petri_matrix = np.zeros((5, 11), dtype=int)
    num_transitions = 5
    modified_matrix = add_missing_connections(petri_matrix, num_transitions)
    # Each transition and place should have at least one connection
    assert np.all(np.sum(modified_matrix[:, :10], axis=0) > 0)
    assert np.all(np.sum(modified_matrix[:, :5], axis=1) > 0)
    assert np.all(np.sum(modified_matrix[:, 5:10], axis=1) > 0)


def test_add_tokens_randomly():
    """Test adding tokens randomly."""
    petri_matrix = np.zeros((5, 11), dtype=int)
    modified_matrix = add_tokens_randomly(petri_matrix)
    assert np.sum(modified_matrix[:, -1]) >= 0
