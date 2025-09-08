import numpy as np
import pytest
from DataGenerate.SPN import is_connected


def test_is_connected_true():
    """Tests that a connected Petri net returns True."""
    petri_matrix = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype="int32",
    )
    assert is_connected(petri_matrix) is True


def test_is_connected_isolated_place():
    """Tests that a Petri net with an isolated place returns False."""
    petri_matrix = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype="int32",
    )
    assert is_connected(petri_matrix) is False


def test_is_connected_isolated_transition():
    """Tests that a Petri net with an isolated transition returns False."""
    petri_matrix = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype="int32",
    )
    # This matrix has a transition (column 2) that is not connected to any place.
    # The sum of column 2 and column 2 + num_transitions is 0.
    # num_transitions = (5 - 1) // 2 = 2
    # pre_sum for transition 2 (index 2) is 0. post_sum for transition 2 (index 4) is 0.
    # This should be caught by the isolated transition check.
    # However, the current implementation of is_connected has a bug.
    # Let's fix it.
    # I will first write a failing test for it, and then fix the bug.
    # The bug is in the check for isolated transitions.
    # It checks `pre_sum + post_sum == 0`, but `pre_sum` and `post_sum` are sums of all transitions.
    # It should check each transition individually.
    # Let's see the implementation again.
    # pre_sum = np.sum(petri_net_matrix[:, :num_transitions], axis=0)
    # post_sum = np.sum(petri_net_matrix[:, num_transitions : 2 * num_transitions], axis=0)
    # if np.any(pre_sum + post_sum == 0):
    # This is actually correct. It checks if any of the transitions has a sum of 0 for its pre and post connections.
    # So, for the matrix above, num_transitions = 2
    # pre_matrix is petri_matrix[:, :2] = [[1, 0], [0, 1]]
    # post_matrix is petri_matrix[:, 2:4] = [[0, 1], [0, 0]]
    # pre_sum is [1, 1]
    # post_sum is [0, 1]
    # pre_sum + post_sum is [1, 2]. np.any([1, 2] == 0) is False. This is correct.
    # The transition I thought was isolated is the second one (index 1).
    # pre_sum[1] is 1, post_sum[1] is 0. Sum is 1. Not isolated.
    # Let's make a truly isolated transition.
    petri_matrix_isolated_transition = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype="int32",
    )
    # num_transitions = 2
    # pre_matrix is [[1, 0], [0, 0]]
    # post_matrix is [[0, 1], [1, 0]]
    # pre_sum is [1, 0]
    # post_sum is [1, 1]
    # pre_sum + post_sum is [2, 1]. No isolated transition.
    # I'm getting confused. Let's create a simpler example.
    # 2 places, 2 transitions.
    # p1 -> t1 -> p2
    # t2 is isolated
    petri_matrix_isolated_t2 = np.array(
        [
            # t1_pre, t2_pre, t1_post, t2_post, markings
            [1, 0, 0, 0, 1],  # p1
            [0, 0, 1, 0, 0],  # p2
        ],
        dtype="int32",
    )
    assert is_connected(petri_matrix_isolated_t2) is False


def test_is_connected_empty():
    """Tests that an empty Petri net returns False."""
    petri_matrix = np.array([], dtype="int32")
    assert is_connected(petri_matrix) is False


def test_is_connected_no_transitions():
    """Tests that a Petri net with no transitions returns False."""
    petri_matrix = np.array([[1]], dtype="int32")
    assert is_connected(petri_matrix) is False


from DataGenerate.PetriGenerate import (
    generate_random_petri_net,
    delete_excess_edges,
    add_missing_connections,
)


def test_generate_random_petri_net():
    """Tests the shape and initial marking of a generated Petri net."""
    num_places = 5
    num_transitions = 10
    petri_matrix = generate_random_petri_net(num_places, num_transitions)
    assert petri_matrix.shape == (num_places, 2 * num_transitions + 1)
    assert np.sum(petri_matrix[:, -1]) == 1


def test_delete_excess_edges():
    """Tests that excess edges are deleted correctly."""
    num_transitions = 2
    petri_matrix = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype="int32",
    )
    pruned_matrix = delete_excess_edges(petri_matrix, num_transitions)
    assert np.sum(pruned_matrix[0, :-1]) == 2

    petri_matrix_t = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype="int32",
    )
    pruned_matrix_t = delete_excess_edges(petri_matrix_t, num_transitions=1)
    assert np.sum(pruned_matrix_t[:, 0]) == 2


def test_add_missing_connections():
    """Tests that missing connections are added correctly."""
    num_transitions = 2
    # Place with no outgoing edges
    petri_matrix_place = np.array(
        [
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype="int32",
    )
    connected_matrix = add_missing_connections(petri_matrix_place, num_transitions)
    assert np.sum(connected_matrix[0, :num_transitions]) > 0

    # Transition with no connections
    petri_matrix_trans = np.array(
        [
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ],
        dtype="int32",
    )
    connected_matrix = add_missing_connections(petri_matrix_trans, num_transitions)
    assert np.sum(connected_matrix[:, 1]) + np.sum(connected_matrix[:, 1 + num_transitions]) > 0
