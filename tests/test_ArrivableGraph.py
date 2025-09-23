import numpy as np
import pytest
from DataGenerate.ArrivableGraph import (
    get_enabled_transitions,
    generate_reachability_graph,
)

@pytest.fixture
def simple_petri_net():
    """A simple Petri net for testing."""
    # P1 -> T1 -> P2
    pre = np.array([[1], [0]])
    post = np.array([[0], [1]])
    initial_marking = np.array([1, 0])
    return np.hstack([pre, post, initial_marking.reshape(-1, 1)])

def test_get_enabled_transitions(simple_petri_net):
    """Test the identification of enabled transitions."""
    pre_matrix = simple_petri_net[:, :1]
    change_matrix = simple_petri_net[:, 1:2] - pre_matrix
    current_marking = simple_petri_net[:, -1]

    new_markings, enabled_transitions = get_enabled_transitions(pre_matrix, change_matrix, current_marking)

    assert len(enabled_transitions) == 1
    assert enabled_transitions[0] == 0
    assert np.allclose(new_markings[0], np.array([0, 1]))

def test_get_enabled_transitions_none():
    """Test when there are no enabled transitions."""
    pre = np.array([[1]])
    post = np.array([[0]])
    current_marking = np.array([0])
    change_matrix = post - pre
    _, enabled_transitions = get_enabled_transitions(pre, change_matrix, current_marking)
    assert len(enabled_transitions) == 0

def test_generate_reachability_graph(simple_petri_net):
    """Test the generation of a reachability graph."""
    v, e, t, n, bounded = generate_reachability_graph(simple_petri_net)

    assert bounded is True
    assert n == 1
    assert len(v) == 2
    assert len(e) == 1
    assert len(t) == 1
    assert np.allclose(v[0], np.array([1, 0]))
    assert np.allclose(v[1], np.array([0, 1]))
    assert e[0] == [0, 1]
    assert t[0] == 0

def test_generate_reachability_graph_unbounded():
    """Test with a Petri net that is unbounded."""
    # T1 -> P1
    pre = np.array([[0]])
    post = np.array([[1]])
    initial_marking = np.array([1])
    petri_net = np.hstack([pre, post, initial_marking.reshape(-1, 1)])

    _, _, _, _, bounded = generate_reachability_graph(petri_net, place_upper_limit=5)

    assert bounded is False

def test_generate_reachability_graph_max_markings():
    """Test that exploration stops when max_markings is reached."""
    # T1 -> P1
    pre = np.array([[0]])
    post = np.array([[1]])
    initial_marking = np.array([1])
    petri_net = np.hstack([pre, post, initial_marking.reshape(-1, 1)])

    v, _, _, _, bounded = generate_reachability_graph(petri_net, max_markings_to_explore=5)

    assert bounded is False
    assert len(v) <= 5

def test_generate_reachability_graph_revisited_marking():
    """Test a graph with a cycle."""
    # P1 -> T1 -> P2 -> T2 -> P1
    pre = np.array([[1, 0], [0, 1]])
    post = np.array([[0, 1], [1, 0]])
    initial_marking = np.array([1, 0])
    petri_net = np.hstack([pre, post, initial_marking.reshape(-1, 1)])

    v, e, t, n, bounded = generate_reachability_graph(petri_net)

    assert bounded is True
    assert n == 2
    assert len(v) == 2
    assert len(e) == 2
    assert len(t) == 2
    assert e[1] == [1, 0] # Should be a back edge
