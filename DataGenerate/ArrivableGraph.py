"""
Generates the reachability graph for a given Petri net definition
using Breadth-First Search (BFS) and optimized marking lookup.
"""

from collections import deque
import numpy as np


def get_enabled_transitions(pre_condition_matrix, change_matrix, current_marking_vector):
    """Identifies enabled transitions and calculates the resulting markings.

    Args:
        pre_condition_matrix (numpy.ndarray): The pre-condition matrix (input arcs).
        change_matrix (numpy.ndarray): The change matrix (Post - Pre).
        current_marking_vector (numpy.ndarray): The current state of the Petri net.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Markings resulting from firing enabled transitions.
            - numpy.ndarray: Indices of the enabled transitions.
    """
    current_marking_expanded = current_marking_vector[:, np.newaxis]
    enabled_transitions = np.where(np.all(current_marking_expanded >= pre_condition_matrix, axis=0))[0]

    if not enabled_transitions.size:
        num_places = pre_condition_matrix.shape[0]
        return np.empty((0, num_places), dtype=int), np.empty((0,), dtype=int)

    new_markings = current_marking_expanded + change_matrix[:, enabled_transitions]
    return new_markings.T, enabled_transitions


def _initialize_bfs(initial_marking):
    """Initializes the data structures for the BFS algorithm."""
    marking_index_counter = 0
    visited_markings_list = [initial_marking]
    explored_markings_dict = {tuple(initial_marking): marking_index_counter}
    processing_queue = deque([marking_index_counter])
    return marking_index_counter, visited_markings_list, explored_markings_dict, processing_queue


def _process_marking(
    current_marking_index,
    visited_markings_list,
    pre_matrix,
    change_matrix,
    place_upper_limit,
    max_markings_to_explore,
):
    """Processes a single marking in the BFS algorithm."""
    current_marking = visited_markings_list[current_marking_index]

    if len(visited_markings_list) >= max_markings_to_explore:
        return None, None, True

    enabled_next_markings, enabled_transition_indices = get_enabled_transitions(
        pre_matrix, change_matrix, current_marking
    )

    if enabled_next_markings.size > 0 and np.any(enabled_next_markings > place_upper_limit):
        return None, None, True

    return enabled_next_markings, enabled_transition_indices, False


def _update_graph(
    new_marking,
    enabled_transition_index,
    current_marking_index,
    marking_index_counter,
    visited_markings_list,
    explored_markings_dict,
    processing_queue,
    reachability_edges,
    edge_transition_indices,
    max_markings_to_explore,
):
    """Updates the reachability graph with a new marking and edge."""
    new_marking_tuple = tuple(new_marking)

    if new_marking_tuple not in explored_markings_dict:
        marking_index_counter += 1
        if marking_index_counter >= max_markings_to_explore:
            reachability_edges.append([current_marking_index, marking_index_counter])
            edge_transition_indices.append(enabled_transition_index)
            return marking_index_counter, True

        visited_markings_list.append(new_marking)
        explored_markings_dict[new_marking_tuple] = marking_index_counter
        processing_queue.append(marking_index_counter)
        reachability_edges.append([current_marking_index, marking_index_counter])
    else:
        existing_index = explored_markings_dict[new_marking_tuple]
        reachability_edges.append([current_marking_index, existing_index])

    edge_transition_indices.append(enabled_transition_index)
    return marking_index_counter, False


def generate_reachability_graph(incidence_matrix_with_initial, place_upper_limit=10, max_markings_to_explore=500):
    """Generates the reachability graph of a Petri net using BFS.

    Args:
        incidence_matrix_with_initial (numpy.ndarray): Petri net definition including
            pre-conditions, post-conditions, and initial marking.
            Format: [pre | post | M0].
        place_upper_limit (int, optional): The upper bound for tokens in any single
            place. Defaults to 10.
        max_markings_to_explore (int, optional): The maximum number of markings to
            explore. Defaults to 500.

    Returns:
        tuple: A tuple containing:
            - list: The list of unique reachable markings (states).
            - list: List of edges [from_marking_idx, to_marking_idx].
            - list: List of transition indices corresponding to each edge.
            - int: Number of transitions in the Petri net.
            - bool: Boolean indicating if the net is bounded.
    """
    incidence_matrix = np.array(incidence_matrix_with_initial)
    num_transitions = incidence_matrix.shape[1] // 2
    pre_matrix = incidence_matrix[:, :num_transitions]
    post_matrix = incidence_matrix[:, num_transitions:-1]
    initial_marking = np.array(incidence_matrix[:, -1], dtype=int)
    change_matrix = post_matrix - pre_matrix

    (
        marking_index_counter,
        visited_markings_list,
        explored_markings_dict,
        processing_queue,
    ) = _initialize_bfs(initial_marking)

    reachability_edges = []
    edge_transition_indices = []
    is_bounded = True

    while processing_queue:
        current_marking_index = processing_queue.popleft()

        enabled_next_markings, enabled_transition_indices, stop_exploration = _process_marking(
            current_marking_index,
            visited_markings_list,
            pre_matrix,
            change_matrix,
            place_upper_limit,
            max_markings_to_explore,
        )

        if stop_exploration:
            is_bounded = False
            break

        if enabled_next_markings is None:
            continue

        for new_marking, enabled_transition_index in zip(enabled_next_markings, enabled_transition_indices):
            marking_index_counter, stop = _update_graph(
                new_marking,
                enabled_transition_index,
                current_marking_index,
                marking_index_counter,
                visited_markings_list,
                explored_markings_dict,
                processing_queue,
                reachability_edges,
                edge_transition_indices,
                max_markings_to_explore,
            )
            if stop:
                is_bounded = False
                break
        if not is_bounded:
            break

    return (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        num_transitions,
        is_bounded,
    )
