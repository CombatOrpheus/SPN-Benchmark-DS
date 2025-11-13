"""
This module generates the reachability graph for a given Petri net definition
using a Breadth-First Search (BFS) algorithm optimized with Numba.
"""

from collections import deque
import numpy as np
import numba
from numba.core import types
from numba.typed import Dict, List
from typing import Tuple, Deque


@numba.jit(nopython=True, cache=True)
def fnv1a_hash(data: np.ndarray) -> np.uint64:
    """Computes the FNV-1a hash of a numpy array.

    Args:
        data: The numpy array to hash.

    Returns:
        The 64-bit FNV-1a hash of the array.
    """
    h = np.uint64(14695981039346656037)
    for i in range(data.shape[0]):
        h ^= np.uint64(data[i])
        h *= np.uint64(1099511628211)
    return h


@numba.jit(nopython=True, cache=True)
def get_enabled_transitions(
    pre_condition_matrix: np.ndarray,
    change_matrix: np.ndarray,
    current_marking_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies enabled transitions and calculates the resulting markings.

    Args:
        pre_condition_matrix: The pre-condition matrix (input arcs).
        change_matrix: The change matrix (Post - Pre).
        current_marking_vector: The current state of the Petri net.

    Returns:
        A tuple containing:
            - Markings resulting from firing enabled transitions.
            - Indices of the enabled transitions.
    """
    num_transitions = pre_condition_matrix.shape[1]
    enabled_mask = np.ones(num_transitions, dtype=np.bool_)
    for t in range(num_transitions):
        for p in range(pre_condition_matrix.shape[0]):
            if current_marking_vector[p] < pre_condition_matrix[p, t]:
                enabled_mask[t] = False
                break
    enabled_transitions = np.where(enabled_mask)[0]

    if not enabled_transitions.size:
        num_places = pre_condition_matrix.shape[0]
        return np.empty((0, num_places), dtype=np.int64), np.empty(
            (0,), dtype=np.int64
        )

    new_markings = (
        current_marking_vector[:, np.newaxis] + change_matrix[:, enabled_transitions]
    )
    return new_markings.T.copy(), enabled_transitions


def _initialize_bfs(
    initial_marking: np.ndarray,
) -> Tuple[int, List, Dict, Deque]:
    """Initializes the data structures for the BFS algorithm.

    Args:
        initial_marking: The initial marking of the Petri net.

    Returns:
        A tuple containing the initial data structures for the BFS algorithm.
    """
    marking_index_counter = 0

    visited_markings_list = List()
    visited_markings_list.append(initial_marking)

    explored_markings_dict = Dict.empty(key_type=types.int64, value_type=types.int64)
    random_vector = np.random.randint(
        1, 1000, size=initial_marking.shape[0], dtype=np.int64
    )
    explored_markings_dict[
        np.dot(initial_marking, random_vector)
    ] = marking_index_counter

    processing_queue = deque([marking_index_counter])

    return (
        marking_index_counter,
        visited_markings_list,
        explored_markings_dict,
        processing_queue,
    )


@numba.jit(nopython=True, cache=True)
def _bfs_core(
    initial_marking: np.ndarray,
    pre_matrix: np.ndarray,
    change_matrix: np.ndarray,
    place_upper_limit: int,
    max_markings_to_explore: int,
) -> Tuple[List, List, List, bool]:
    """Core BFS loop optimized with Numba.

    Args:
        initial_marking: The initial marking of the Petri net.
        pre_matrix: The pre-condition matrix.
        change_matrix: The change matrix.
        place_upper_limit: The upper bound for tokens in any single place.
        max_markings_to_explore: The maximum number of markings to explore.

    Returns:
        A tuple containing the results of the BFS algorithm.
    """
    marking_index_counter = 0

    visited_markings_list = List()
    visited_markings_list.append(initial_marking)

    explored_markings_dict = Dict.empty(
        key_type=types.uint64, value_type=types.int64
    )
    explored_markings_dict[fnv1a_hash(initial_marking)] = marking_index_counter

    queue = np.empty(max_markings_to_explore, dtype=np.int64)
    queue[0] = marking_index_counter
    head = 0
    tail = 1

    reachability_edges = List()
    edge_transition_indices = List()
    is_bounded = True

    while head < tail:
        current_marking_index = queue[head]
        head += 1
        current_marking = visited_markings_list[current_marking_index]

        if len(visited_markings_list) >= max_markings_to_explore:
            is_bounded = False
            break

        enabled_next_markings, enabled_transition_indices = get_enabled_transitions(
            pre_matrix, change_matrix, current_marking
        )

        if enabled_next_markings.size > 0 and np.any(
            enabled_next_markings > place_upper_limit
        ):
            is_bounded = False
            break

        for i in range(enabled_next_markings.shape[0]):
            new_marking = enabled_next_markings[i]
            enabled_transition_index = enabled_transition_indices[i]
            new_marking_hash = fnv1a_hash(new_marking)

            if new_marking_hash not in explored_markings_dict:
                marking_index_counter += 1
                if marking_index_counter >= max_markings_to_explore:
                    reachability_edges.append(
                        (current_marking_index, marking_index_counter)
                    )
                    edge_transition_indices.append(enabled_transition_index)
                    is_bounded = False
                    break

                visited_markings_list.append(new_marking)
                explored_markings_dict[new_marking_hash] = marking_index_counter
                queue[tail] = marking_index_counter
                tail += 1
                reachability_edges.append(
                    (current_marking_index, marking_index_counter)
                )
            else:
                existing_index = explored_markings_dict[new_marking_hash]
                reachability_edges.append((current_marking_index, existing_index))

            edge_transition_indices.append(enabled_transition_index)
        if not is_bounded:
            break

    return (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        is_bounded,
    )


def generate_reachability_graph(
    incidence_matrix_with_initial: np.ndarray,
    place_upper_limit: int = 10,
    max_markings_to_explore: int = 500,
) -> Tuple[List[np.ndarray], List[List[int]], List[int], int, bool]:
    """Generates the reachability graph of a Petri net using BFS.

    Args:
        incidence_matrix_with_initial: Petri net definition including
            pre-conditions, post-conditions, and initial marking.
            Format: [pre | post | M0].
        place_upper_limit: The upper bound for tokens in any single
            place. Defaults to 10.
        max_markings_to_explore: The maximum number of markings to
            explore. Defaults to 500.

    Returns:
        A tuple containing:
            - The list of unique reachable markings (states).
            - List of edges [from_marking_idx, to_marking_idx].
            - List of transition indices corresponding to each edge.
            - Number of transitions in the Petri net.
            - Boolean indicating if the net is bounded.
    """
    incidence_matrix = np.array(incidence_matrix_with_initial)
    num_transitions = incidence_matrix.shape[1] // 2
    pre_matrix = incidence_matrix[:, :num_transitions]
    post_matrix = incidence_matrix[:, num_transitions:-1]
    initial_marking = np.array(incidence_matrix[:, -1], dtype=np.int64)
    change_matrix = post_matrix - pre_matrix

    (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        is_bounded,
    ) = _bfs_core(
        initial_marking,
        pre_matrix,
        change_matrix,
        place_upper_limit,
        max_markings_to_explore,
    )

    py_visited_markings_list = [np.array(m) for m in visited_markings_list]
    py_reachability_edges = [list(e) for e in reachability_edges]
    py_edge_transition_indices = list(edge_transition_indices)

    return (
        py_visited_markings_list,
        py_reachability_edges,
        py_edge_transition_indices,
        num_transitions,
        is_bounded,
    )
