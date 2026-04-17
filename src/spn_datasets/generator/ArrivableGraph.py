"""
Generates the reachability graph for a given Petri net definition
using Breadth-First Search (BFS) and optimized marking lookup.
"""

import numpy as np
import numba
from numba.core import types
from numba.typed import Dict


@numba.jit(nopython=True, cache=True)
def fnv1a_hash(data):
    """FNV-1a hash function for a numpy array."""
    h = np.uint64(14695981039346656037)
    for i in range(data.shape[0]):
        h ^= np.uint64(data[i])
        h *= np.uint64(1099511628211)
    return h


@numba.jit(nopython=True, cache=True)
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
    num_places = pre_condition_matrix.shape[0]
    num_transitions = pre_condition_matrix.shape[1]

    # Pre-allocate array to avoid intermediate boolean mask and np.where
    enabled_transitions = np.empty(num_transitions, dtype=np.int64)
    enabled_count = 0

    for t in range(num_transitions):
        is_enabled = True
        for p in range(num_places):
            if current_marking_vector[p] < pre_condition_matrix[p, t]:
                is_enabled = False
                break
        if is_enabled:
            enabled_transitions[enabled_count] = t
            enabled_count += 1

    if enabled_count == 0:
        return np.empty((0, num_places), dtype=np.int64), np.empty((0,), dtype=np.int64)

    enabled_transitions = enabled_transitions[:enabled_count]

    # Pre-allocate new_markings to avoid implicit advanced indexing allocation
    new_markings = np.empty((enabled_count, num_places), dtype=np.int64)

    for i in range(enabled_count):
        t = enabled_transitions[i]
        for p in range(num_places):
            new_markings[i, p] = current_marking_vector[p] + change_matrix[p, t]

    return new_markings, enabled_transitions


@numba.jit(nopython=True, cache=True)
def _bfs_core(
    initial_marking,
    pre_matrix,
    change_matrix,
    place_upper_limit,
    max_markings_to_explore,
):
    """Core BFS loop optimized with Numba."""
    marking_index_counter = 0

    # Visited markings are stored in a dense numpy array for efficient access
    # Since we know max_markings_to_explore, we can pre-allocate
    num_places = initial_marking.shape[0]
    visited_markings_array = np.empty((max_markings_to_explore, num_places), dtype=np.int64)
    visited_markings_array[0] = initial_marking

    # Explored markings are stored in a Numba Dict for fast lookups using a hash as a key
    explored_markings_dict = Dict.empty(key_type=types.uint64, value_type=types.int64)
    explored_markings_dict[fnv1a_hash(initial_marking)] = marking_index_counter

    # The processing queue for the BFS algorithm, using a circular queue
    queue = np.empty(max_markings_to_explore, dtype=np.int64)
    queue[0] = marking_index_counter
    head = 0
    tail = 1

    num_transitions = pre_matrix.shape[1]

    # Data structures to store the graph
    reachability_edges_src = np.empty(max_markings_to_explore * num_transitions, dtype=np.int64)
    reachability_edges_dst = np.empty(max_markings_to_explore * num_transitions, dtype=np.int64)
    edge_transition_indices = np.empty(max_markings_to_explore * num_transitions, dtype=np.int64)
    edge_count = 0
    is_bounded = True

    while head < tail:
        current_marking_index = queue[head]
        head += 1
        current_marking = visited_markings_array[current_marking_index]

        if marking_index_counter >= max_markings_to_explore - 1:
            is_bounded = False
            break

        enabled_next_markings, enabled_transition_indices = get_enabled_transitions(
            pre_matrix, change_matrix, current_marking
        )

        # ⚡ Bolt Optimization: Replace `np.any(enabled_next_markings > place_upper_limit)`
        # with explicit nested loops. Numba compiles NumPy high-level reduction operators
        # by first allocating the intermediate boolean mask and evaluating it entirely
        # before reducing. Explicit loops avoid allocation and support early exit.
        if enabled_next_markings.size > 0:
            exceeds_limit = False
            for i in range(enabled_next_markings.shape[0]):
                for j in range(enabled_next_markings.shape[1]):
                    if enabled_next_markings[i, j] > place_upper_limit:
                        exceeds_limit = True
                        break
                if exceeds_limit:
                    break
            if exceeds_limit:
                is_bounded = False
                break

        for i in range(enabled_next_markings.shape[0]):
            new_marking = enabled_next_markings[i]
            enabled_transition_index = enabled_transition_indices[i]
            new_marking_hash = fnv1a_hash(new_marking)

            if new_marking_hash not in explored_markings_dict:
                marking_index_counter += 1
                visited_markings_array[marking_index_counter] = new_marking
                explored_markings_dict[new_marking_hash] = marking_index_counter

                if marking_index_counter >= max_markings_to_explore - 1:
                    reachability_edges_src[edge_count] = current_marking_index
                    reachability_edges_dst[edge_count] = marking_index_counter
                    edge_transition_indices[edge_count] = enabled_transition_index
                    edge_count += 1
                    is_bounded = False
                    break

                queue[tail] = marking_index_counter
                tail += 1

                reachability_edges_src[edge_count] = current_marking_index
                reachability_edges_dst[edge_count] = marking_index_counter
                edge_transition_indices[edge_count] = enabled_transition_index
                edge_count += 1
            else:
                existing_index = explored_markings_dict[new_marking_hash]
                reachability_edges_src[edge_count] = current_marking_index
                reachability_edges_dst[edge_count] = existing_index
                edge_transition_indices[edge_count] = enabled_transition_index
                edge_count += 1

        if not is_bounded:
            break

    return (
        visited_markings_array[: marking_index_counter + 1],
        reachability_edges_src[:edge_count],
        reachability_edges_dst[:edge_count],
        edge_transition_indices[:edge_count],
        is_bounded,
    )


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
            - numpy.ndarray: The unique reachable markings (states).
            - numpy.ndarray: Edges as [from_marking_idx, to_marking_idx].
            - numpy.ndarray: Transition indices corresponding to each edge.
            - int: Number of transitions in the Petri net.
            - bool: Boolean indicating if the net is bounded.
    """
    incidence_matrix = np.asarray(incidence_matrix_with_initial)
    num_transitions = incidence_matrix.shape[1] // 2
    pre_matrix = incidence_matrix[:, :num_transitions]
    post_matrix = incidence_matrix[:, num_transitions:-1]
    initial_marking = np.asarray(incidence_matrix[:, -1], dtype=np.int64)
    change_matrix = post_matrix - pre_matrix

    visited_markings_list, reach_src, reach_dst, edge_transition_indices, is_bounded = _bfs_core(
        initial_marking,
        pre_matrix,
        change_matrix,
        place_upper_limit,
        max_markings_to_explore,
    )

    reachability_edges = np.column_stack((reach_src, reach_dst))

    return (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        num_transitions,
        is_bounded,
    )
