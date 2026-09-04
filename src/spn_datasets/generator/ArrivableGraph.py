"""
Generates the reachability graph for a given Petri net definition
using Breadth-First Search (BFS) and optimized marking lookup.
"""

import numpy as np
import numba

import threading

# Thread-local storage for scratchpad buffers to avoid allocations per SPN
_scratchpad = threading.local()


def _get_scratchpad(max_markings, num_places, num_transitions):
    key = (max_markings, num_places, num_transitions)

    if not hasattr(_scratchpad, "cache"):
        _scratchpad.cache = {}

    if key not in _scratchpad.cache:
        visited = np.empty((max_markings, num_places), dtype=np.int64)
        queue = np.empty(max_markings, dtype=np.int64)
        max_edges = max_markings * num_transitions
        reach_src = np.empty(max_edges, dtype=np.int64)
        reach_dst = np.empty(max_edges, dtype=np.int64)
        edge_indices = np.empty(max_edges, dtype=np.int64)
        enabled_trans = np.empty(num_transitions, dtype=np.int64)
        new_marks = np.empty((num_transitions, num_places), dtype=np.int64)

        # Flat open-addressing hash table with load factor <= 0.5
        table_capacity = 1 << int(np.ceil(np.log2(max(16, max_markings * 2))))
        table_keys = np.empty(table_capacity, dtype=np.uint64)
        table_values = np.full(table_capacity, -1, dtype=np.int64)
        used_slots = np.empty(max_markings + 1, dtype=np.int64)
        num_used_ptr = np.zeros(1, dtype=np.int64)

        _scratchpad.cache[key] = (
            visited,
            queue,
            reach_src,
            reach_dst,
            edge_indices,
            enabled_trans,
            new_marks,
            table_keys,
            table_values,
            used_slots,
            num_used_ptr,
        )

    return _scratchpad.cache[key]


@numba.jit(nopython=True, cache=True)
def fnv1a_hash(data):
    """FNV-1a hash function for a numpy array."""
    h = np.uint64(14695981039346656037)
    for i in range(data.shape[0]):
        h ^= np.uint64(data[i])
        h *= np.uint64(1099511628211)
    return h


@numba.jit(nopython=True, cache=True)
def get_enabled_transitions(
    pre_condition_matrix, change_matrix, current_marking_vector, enabled_transitions, new_markings
):
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
    # enabled_transitions is pre-allocated
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
        return 0

    # enabled_transitions is sliced outside

    # Pre-allocate new_markings to avoid implicit advanced indexing allocation
    # new_markings is pre-allocated (num_transitions, num_places)

    for i in range(enabled_count):
        t = enabled_transitions[i]
        for p in range(num_places):
            new_markings[i, p] = current_marking_vector[p] + change_matrix[p, t]

    return enabled_count


@numba.jit(nopython=True, cache=True)
def _lookup_or_insert(
    table_keys,
    table_values,
    used_slots,
    num_used_ptr,
    visited_markings,
    new_marking,
    key_hash,
    table_mask,
    num_places,
):
    """Probes the flat open-addressing table with exact marking comparison on hash matches.

    Returns:
        tuple (existing_index, slot):
            - If found: (existing_index >= 0, slot)
            - If empty: (-1, slot), with key_hash inserted and slot tracked in used_slots.
    """
    slot = key_hash & table_mask
    one = np.uint64(1)
    while True:
        existing_idx = table_values[slot]
        if existing_idx == -1:
            u_idx = num_used_ptr[0]
            used_slots[u_idx] = slot
            num_used_ptr[0] = u_idx + 1
            table_keys[slot] = key_hash
            return -1, slot
        elif table_keys[slot] == key_hash:
            match = True
            for p in range(num_places):
                if visited_markings[existing_idx, p] != new_marking[p]:
                    match = False
                    break
            if match:
                return existing_idx, slot
        slot = (slot + one) & table_mask


@numba.jit(nopython=True, cache=True)
def _bfs_core(
    initial_marking,
    pre_matrix,
    change_matrix,
    place_upper_limit,
    max_markings_to_explore,
    visited_markings_array,
    queue,
    reachability_edges_src,
    reachability_edges_dst,
    edge_transition_indices,
    scratch_enabled_transitions,
    scratch_new_markings,
    table_keys,
    table_values,
    used_slots,
    num_used_ptr,
    table_mask,
):
    """Core BFS loop optimized with Numba and flat open-addressing hash table."""
    num_places = initial_marking.shape[0]
    num_used_ptr[0] = 0

    marking_index_counter = 0
    visited_markings_array[0] = initial_marking

    initial_hash = fnv1a_hash(initial_marking)
    _, init_slot = _lookup_or_insert(
        table_keys,
        table_values,
        used_slots,
        num_used_ptr,
        visited_markings_array,
        initial_marking,
        initial_hash,
        table_mask,
        num_places,
    )
    table_values[init_slot] = marking_index_counter

    queue[0] = marking_index_counter
    head = 0
    tail = 1

    num_transitions = pre_matrix.shape[1]
    edge_count = 0
    is_bounded = True

    while head < tail:
        current_marking_index = queue[head]
        head += 1
        current_marking = visited_markings_array[current_marking_index]

        if marking_index_counter >= max_markings_to_explore - 1:
            is_bounded = False
            break

        enabled_count = get_enabled_transitions(
            pre_matrix, change_matrix, current_marking, scratch_enabled_transitions, scratch_new_markings
        )
        enabled_next_markings = scratch_new_markings[:enabled_count]
        enabled_transition_indices = scratch_enabled_transitions[:enabled_count]

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

            existing_index, slot = _lookup_or_insert(
                table_keys,
                table_values,
                used_slots,
                num_used_ptr,
                visited_markings_array,
                new_marking,
                new_marking_hash,
                table_mask,
                num_places,
            )

            if existing_index == -1:
                marking_index_counter += 1
                visited_markings_array[marking_index_counter] = new_marking
                table_values[slot] = marking_index_counter

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
                reachability_edges_src[edge_count] = current_marking_index
                reachability_edges_dst[edge_count] = existing_index
                edge_transition_indices[edge_count] = enabled_transition_index
                edge_count += 1

        if not is_bounded:
            break

    # Fast reset of modified table slots
    for k in range(num_used_ptr[0]):
        table_values[used_slots[k]] = -1
    num_used_ptr[0] = 0

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

    num_places = initial_marking.shape[0]
    (
        scratch_visited,
        scratch_queue,
        scratch_reach_src,
        scratch_reach_dst,
        scratch_edge_indices,
        scratch_enabled_trans,
        scratch_new_marks,
        scratch_table_keys,
        scratch_table_values,
        scratch_used_slots,
        scratch_num_used_ptr,
    ) = _get_scratchpad(max_markings_to_explore, num_places, num_transitions)

    table_mask = np.uint64(scratch_table_keys.shape[0] - 1)

    visited_markings_list, reach_src, reach_dst, edge_transition_indices, is_bounded = _bfs_core(
        initial_marking,
        pre_matrix,
        change_matrix,
        place_upper_limit,
        max_markings_to_explore,
        scratch_visited,
        scratch_queue,
        scratch_reach_src,
        scratch_reach_dst,
        scratch_edge_indices,
        scratch_enabled_trans,
        scratch_new_marks,
        scratch_table_keys,
        scratch_table_values,
        scratch_used_slots,
        scratch_num_used_ptr,
        table_mask,
    )

    # Note: visited_markings_list, reach_src, etc. are views of the scratchpad.
    # To prevent issues if these arrays are modified later while the scratchpad is reused,
    # we copy them out.
    visited_markings_list = visited_markings_list.copy()
    reach_src = reach_src.copy()
    reach_dst = reach_dst.copy()
    edge_transition_indices = edge_transition_indices.copy()

    reachability_edges = np.column_stack((reach_src, reach_dst))

    return (
        visited_markings_list,
        reachability_edges,
        edge_transition_indices,
        num_transitions,
        is_bounded,
    )
