import numpy as np
import pytest
from DataGenerate.ArrivableGraph import get_enabled_transitions, generate_reachability_graph


@pytest.fixture
def simple_petri_net():
    """A simple Petri net: p1 -> t1 -> p2"""
    pre = np.array([[1], [0]])
    post = np.array([[0], [1]])
    change = post - pre
    return pre, change


def test_get_enabled_transitions(simple_petri_net):
    """Tests the identification of enabled transitions."""
    pre, change = simple_petri_net
    # Marking where t1 is enabled
    marking_enabled = np.array([1, 0])
    next_markings, enabled_trans = get_enabled_transitions(pre, change, marking_enabled)
    assert len(enabled_trans) == 1
    assert enabled_trans[0] == 0
    assert np.array_equal(next_markings[0], np.array([0, 1]))

    # Marking where t1 is not enabled
    marking_disabled = np.array([0, 1])
    next_markings, enabled_trans = get_enabled_transitions(pre, change, marking_disabled)
    assert len(enabled_trans) == 0
    assert next_markings.shape == (0, 2)


def test_generate_reachability_graph_simple():
    """Tests the reachability graph generation for a simple bounded net."""
    # p1 -> t1 -> p2
    incidence_matrix = np.array([[1, 0, 1], [0, 1, 0]])  # pre | post | M0
    vertices, edges, transitions, num_trans, is_bounded = generate_reachability_graph(incidence_matrix)
    assert is_bounded is True
    assert num_trans == 1
    assert len(vertices) == 2
    assert np.array_equal(vertices[0], np.array([1, 0]))
    assert np.array_equal(vertices[1], np.array([0, 1]))
    assert len(edges) == 1
    assert edges[0] == [0, 1]
    assert len(transitions) == 1
    assert transitions[0] == 0


def test_generate_reachability_graph_unbounded():
    """Tests the reachability graph generation for an unbounded net."""
    # t1 -> p1 -> t1 (unbounded)
    incidence_matrix = np.array(
        [
            [0, 1, 1],
        ]
    )
    _, _, _, _, is_bounded = generate_reachability_graph(incidence_matrix, place_upper_limit=5)
    assert is_bounded is False


def test_generate_reachability_graph_max_markings():
    """Tests the reachability graph generation with a max markings limit."""
    # p1 -> t1 -> p2 -> t2 -> p3 ... (a long chain)
    num_places = 10
    num_transitions = 9
    pre = np.zeros((num_places, num_transitions))
    post = np.zeros((num_places, num_transitions))
    for i in range(num_transitions):
        pre[i, i] = 1
        post[i + 1, i] = 1
    m0 = np.zeros((num_places, 1))
    m0[0] = 1
    incidence_matrix = np.hstack([pre, post, m0])

    max_markings = 5
    vertices, _, _, _, is_bounded = generate_reachability_graph(incidence_matrix, max_markings_to_explore=max_markings)
    assert is_bounded is False
    assert len(vertices) == max_markings
