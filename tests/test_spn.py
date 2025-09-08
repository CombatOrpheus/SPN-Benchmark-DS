import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix
from DataGenerate.SPN import (
    compute_state_equation,
    solve_for_steady_state,
    compute_average_markings,
)


@pytest.fixture
def simple_spn():
    """A simple SPN with 2 places, 1 transition, and a known steady state."""
    # p1 -> t1 -> p2
    # Initial marking: p1=1, p2=0. Rate(t1)=1.0
    # Reachable states:
    # s0 = (1, 0)
    # s1 = (0, 1)
    # Transition s0 -> s1 with rate 1.0
    vertices = [np.array([1, 0]), np.array([0, 1])]
    edges = [[0, 1]]
    arc_transitions = [0]
    lambda_values = np.array([1.0])

    # Ax = 0, where x is the steady state probability vector
    # -1.0 * x0 + 0.0 * x1 = 0
    #  1.0 * x0 + 0.0 * x1 = 0
    # This is wrong.
    # The state transition matrix Q should be:
    #      s0   s1
    # s0 [-1.0, 0.0]
    # s1 [ 1.0, 0.0]
    # And Qx = 0
    # -x0 = 0
    # x0 = 0
    # x0 + x1 = 1 => x1 = 1
    # So steady state is [0, 1]
    # Let's make it a bit more interesting, with a transition back.
    # p1 -> t1 -> p2 -> t2 -> p1
    # Initial marking: p1=1, p2=0. Rate(t1)=1.0, Rate(t2)=1.0
    # Reachable states: s0=(1,0), s1=(0,1)
    # s0 -> s1 with rate 1.0 (t1)
    # s1 -> s0 with rate 1.0 (t2)
    vertices = [np.array([1, 0]), np.array([0, 1])]
    edges = [[0, 1], [1, 0]]
    arc_transitions = [0, 1]
    lambda_values = np.array([1.0, 1.0])
    # Qx = 0
    #      s0   s1
    # s0 [-1.0, 1.0]
    # s1 [ 1.0,-1.0]
    # -x0 + x1 = 0 => x0 = x1
    # x0 + x1 = 1 => 2x0 = 1 => x0 = 0.5
    # Steady state: [0.5, 0.5]
    expected_steady_state = np.array([0.5, 0.5])

    return vertices, edges, arc_transitions, lambda_values, expected_steady_state


def test_compute_state_equation(simple_spn):
    """Tests the computation of the state equation for a simple SPN."""
    vertices, edges, arc_transitions, lambda_values, _ = simple_spn
    state_matrix, target_vector = compute_state_equation(vertices, edges, arc_transitions, lambda_values)
    assert isinstance(state_matrix, (csc_array, csc_matrix))
    assert state_matrix.shape == (len(vertices) + 1, len(vertices))
    assert target_vector.shape == (len(vertices) + 1,)
    assert target_vector[-1] == 1.0
    # Manually check a few values
    # state_matrix[s0, s0] = -lambda(t1) = -1.0
    assert state_matrix[0, 0] == -1.0
    # state_matrix[s1, s0] = lambda(t1) = 1.0
    assert state_matrix[1, 0] == 1.0
    # state_matrix[s0, s1] = lambda(t2) = 1.0
    assert state_matrix[0, 1] == 1.0
    # state_matrix[s1, s1] = -lambda(t2) = -1.0
    assert state_matrix[1, 1] == -1.0


def test_solve_for_steady_state(simple_spn):
    """Tests the steady state solver for a simple SPN."""
    vertices, edges, arc_transitions, lambda_values, expected_steady_state = simple_spn
    state_matrix, target_vector = compute_state_equation(vertices, edges, arc_transitions, lambda_values)
    steady_state_probs = solve_for_steady_state(state_matrix, target_vector)
    assert steady_state_probs is not None
    assert np.allclose(steady_state_probs, expected_steady_state, atol=1e-6)


def test_compute_average_markings(simple_spn):
    """Tests the computation of average markings."""
    vertices, _, _, _, expected_steady_state = simple_spn
    vertices_np = np.array(vertices)
    # avg_tokens_p1 = 1*0.5 + 0*0.5 = 0.5
    # avg_tokens_p2 = 0*0.5 + 1*0.5 = 0.5
    expected_avg_markings = np.array([0.5, 0.5])

    _, avg_markings = compute_average_markings(vertices_np, expected_steady_state)
    assert np.allclose(avg_markings, expected_avg_markings, atol=1e-6)
