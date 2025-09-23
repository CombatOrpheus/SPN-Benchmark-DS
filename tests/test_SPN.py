import numpy as np
import pytest
from scipy.sparse import csc_array
from DataGenerate.SPN import (
    is_connected,
    compute_state_equation,
    solve_for_steady_state,
    compute_average_markings,
    generate_stochastic_net_task,
    filter_spn,
    get_spn_info,
    _compute_state_equation_numba,
)


def test_is_connected_positive():
    """Test that a connected Petri net is correctly identified."""
    # 2 places, 2 transitions, last column is initial marking
    petri_net_matrix = np.array([[1, 0, 0, 1, 1], [0, 1, 1, 0, 0]])  # P1: T1 -> P1, T2 -> P1  # P2: T2 -> P2
    assert is_connected(petri_net_matrix) is True


def test_is_connected_isolated_place():
    """Test that a Petri net with an isolated place is correctly identified."""
    petri_net_matrix = np.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])  # Isolated place
    assert is_connected(petri_net_matrix) is False


def test_is_connected_isolated_transition():
    """Test that a Petri net with an isolated transition is correctly identified."""
    # 2 places, 3 transitions, last column is initial marking
    # T2 is isolated
    petri_net_matrix = np.array(
        [[1, 0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0]]  # P1: T1 -> P1, T3 -> P1  # P2: T3 -> P2
    )
    assert is_connected(petri_net_matrix) is False


def test_is_connected_empty():
    """Test that an empty Petri net is correctly identified as not connected."""
    petri_net_matrix = np.array([])
    assert is_connected(petri_net_matrix) is False


def test_is_connected_single_place_no_transitions():
    """Test a Petri net with a single place and no transitions."""
    # 1 place, 0 transitions, last column is initial marking
    petri_net_matrix = np.array([[1]])
    assert is_connected(petri_net_matrix) is False


def test_compute_state_equation():
    """Test the computation of the state equation."""
    vertices = [np.array([1, 0]), np.array([0, 1])]
    edges = [[0, 1]]
    arc_transitions = [0]
    lambda_values = np.array([2.0])
    state_matrix, target_vector = compute_state_equation(vertices, edges, arc_transitions, lambda_values)

    assert state_matrix.shape == (3, 2)
    assert target_vector.shape == (3,)
    # expected state matrix:
    # [[-2, 0],
    #  [2, 0],
    #  [1, 1]]
    assert np.allclose(state_matrix.toarray(), np.array([[-2.0, 0.0], [2.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(target_vector, np.array([0.0, 0.0, 1.0]))


def test_solve_for_steady_state():
    """Test solving for steady-state probabilities."""
    state_matrix = np.array([[-2.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    state_matrix_sparse = csc_array(state_matrix)
    target_vector = np.array([0.0, 0.0, 1.0])
    probs = solve_for_steady_state(state_matrix_sparse, target_vector)
    assert probs is not None
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0)


def test_solve_for_steady_state_no_solution():
    """Test solve_for_steady_state when no solution is found."""
    # This matrix is singular, which should cause lsmr to fail to converge
    state_matrix = csc_array(np.array([[0.0, 0.0], [0.0, 0.0]]))
    target_vector = np.array([1.0, 1.0])
    probs = solve_for_steady_state(state_matrix, target_vector)
    assert probs is None


def test_compute_average_markings():
    """Test the calculation of average markings."""
    vertices = np.array([[1, 0], [0, 1], [1, 1]])
    steady_state_probs = np.array([0.25, 0.25, 0.5])
    _, avg_markings = compute_average_markings(vertices, steady_state_probs)
    assert np.allclose(avg_markings, np.array([0.75, 0.75]))


def test_generate_stochastic_net_task():
    """Test the generation of a stochastic net task."""
    vertices = [np.array([1, 0]), np.array([0, 1])]
    edges = [[0, 1]]
    arc_transitions = [0]
    num_transitions = 1
    _, _, markings, _, success = generate_stochastic_net_task(vertices, edges, arc_transitions, num_transitions)
    assert success is True
    assert markings is not None


def test_filter_spn_unconnected():
    """Test filter_spn with an unconnected petri net."""
    petri_net_matrix = np.array([[1]])
    _, success = filter_spn(petri_net_matrix)
    assert success is False


def test_get_spn_info_unconnected():
    """Test get_spn_info with an unconnected petri net."""
    petri_net_matrix = np.array([[1]])
    _, success = get_spn_info(petri_net_matrix, [], [], [], np.array([]))
    assert success is False


def test_compute_state_equation_numba():
    """Test the Numba-optimized state equation computation."""
    num_vertices = 2
    edges = np.array([[0, 1]])
    arc_transitions = np.array([0])
    lambda_values = np.array([2.0])
    state_matrix = _compute_state_equation_numba(num_vertices, edges, arc_transitions, lambda_values)
    expected_matrix = np.array([[-2.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    assert np.allclose(state_matrix, expected_matrix)
