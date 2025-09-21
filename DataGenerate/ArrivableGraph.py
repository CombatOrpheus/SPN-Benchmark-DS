"""
Generates the reachability graph for a given Petri net definition
using Breadth-First Search (BFS) and optimized marking lookup.
"""

from collections import deque
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def get_enabled_transitions_numba(pre_condition_matrix, change_matrix, current_marking_vector):
    num_places, num_transitions = pre_condition_matrix.shape
    enabled_transitions_mask = np.ones(num_transitions, dtype=np.bool_)

    for p in range(num_places):
        for t in range(num_transitions):
            if enabled_transitions_mask[t]:
                if current_marking_vector[p] < pre_condition_matrix[p, t]:
                    enabled_transitions_mask[t] = False

    enabled_transitions = np.where(enabled_transitions_mask)[0]

    if not enabled_transitions.size:
        return np.empty((0, num_places), dtype=np.int64), np.empty((0,), dtype=np.int64)

    new_markings = current_marking_vector.copy().reshape(-1, 1) + change_matrix[:, enabled_transitions]
    return new_markings.T, enabled_transitions


@jit(nopython=True, cache=True)
def _bfs_core_numba(
    pre_matrix, change_matrix, initial_marking, place_upper_limit, max_markings_to_explore
):
    visited_markings_list = [initial_marking]
    # Numba doesn't support dicts of tuples as keys well, so we use a list of arrays
    # and check for containment manually. This is slow, but for many problems the
    # state space is small enough that it's not a bottleneck.
    explored_markings_list = [initial_marking]

    q = [0]
    head = 0

    reachability_edges = []
    edge_transition_indices = []
    is_bounded = True

    while head < len(q):
        current_marking_index = q[head]
        head += 1
        current_marking = visited_markings_list[current_marking_index]

        if len(visited_markings_list) >= max_markings_to_explore:
            is_bounded = False
            break

        enabled_next_markings, enabled_transition_indices = get_enabled_transitions_numba(
            pre_matrix, change_matrix, current_marking
        )

        if enabled_next_markings.shape[0] > 0 and np.any(enabled_next_markings > place_upper_limit):
            is_bounded = False
            break

        for i in range(enabled_next_markings.shape[0]):
            new_marking = enabled_next_markings[i].copy()
            enabled_transition_index = enabled_transition_indices[i]

            # Manual check for containment
            is_explored = False
            existing_index = -1
            for j, explored in enumerate(explored_markings_list):
                if np.array_equal(new_marking, explored):
                    is_explored = True
                    existing_index = j
                    break

            if not is_explored:
                marking_index_counter = len(visited_markings_list)
                if marking_index_counter >= max_markings_to_explore:
                    reachability_edges.append(np.array([current_marking_index, marking_index_counter]))
                    edge_transition_indices.append(enabled_transition_index)
                    is_bounded = False
                    break

                visited_markings_list.append(new_marking)
                explored_markings_list.append(new_marking)
                q.append(marking_index_counter)
                reachability_edges.append(np.array([current_marking_index, marking_index_counter]))
            else:
                reachability_edges.append(np.array([current_marking_index, existing_index]))

            edge_transition_indices.append(enabled_transition_index)
        if not is_bounded:
            break

    return visited_markings_list, reachability_edges, edge_transition_indices, is_bounded


def generate_reachability_graph(incidence_matrix_with_initial, place_upper_limit=10, max_markings_to_explore=500):
    incidence_matrix = np.array(incidence_matrix_with_initial)
    num_transitions = incidence_matrix.shape[1] // 2
    pre_matrix = incidence_matrix[:, :num_transitions]
    post_matrix = incidence_matrix[:, num_transitions:-1]
    initial_marking = np.array(incidence_matrix[:, -1], dtype=int)
    change_matrix = post_matrix - pre_matrix

    (
        visited_markings_list,
        reachability_edges_np,
        edge_transition_indices_np,
        is_bounded,
    ) = _bfs_core_numba(
        pre_matrix, change_matrix, initial_marking, place_upper_limit, max_markings_to_explore
    )

    # Convert back to lists of lists/ints for compatibility
    reachability_edges = [edge.tolist() for edge in reachability_edges_np]
    edge_transition_indices = edge_transition_indices_np

    return (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        num_transitions,
        is_bounded,
    )
