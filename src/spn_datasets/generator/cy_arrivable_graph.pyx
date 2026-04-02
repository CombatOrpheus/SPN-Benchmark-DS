# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair

# Initialize numpy C-API
np.import_array()


cdef inline unsigned long long fnv1a_hash(long[:] data) nogil:
    """FNV-1a hash function for a 1D memoryview."""
    cdef unsigned long long h = 14695981039346656037ULL
    cdef Py_ssize_t i
    cdef Py_ssize_t n = data.shape[0]

    for i in range(n):
        h ^= <unsigned long long>data[i]
        h *= 1099511628211ULL
    return h


@cython.boundscheck(False)
@cython.wraparound(False)
def get_enabled_transitions_cython(
    int[:, :] pre_condition_matrix,
    int[:, :] change_matrix,
    long[:] current_marking_vector
):
    """Identifies enabled transitions and calculates the resulting markings."""
    cdef Py_ssize_t num_places = pre_condition_matrix.shape[0]
    cdef Py_ssize_t num_transitions = pre_condition_matrix.shape[1]

    cdef vector[long] enabled_transitions
    cdef Py_ssize_t t, p, i
    cdef bint is_enabled

    # Find enabled transitions
    for t in range(num_transitions):
        is_enabled = True
        for p in range(num_places):
            if current_marking_vector[p] < pre_condition_matrix[p, t]:
                is_enabled = False
                break
        if is_enabled:
            enabled_transitions.push_back(t)

    cdef Py_ssize_t enabled_count = enabled_transitions.size()

    if enabled_count == 0:
        return np.empty((0, num_places), dtype=np.int64), np.empty((0,), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] new_markings_np = np.empty((enabled_count, num_places), dtype=np.int64)
    cdef long[:, :] new_markings = new_markings_np

    cdef np.ndarray[np.int64_t, ndim=1] enabled_transitions_np = np.empty(enabled_count, dtype=np.int64)
    cdef long[:] enabled_transitions_view = enabled_transitions_np

    for i in range(enabled_count):
        t = enabled_transitions[i]
        enabled_transitions_view[i] = t
        for p in range(num_places):
            new_markings[i, p] = current_marking_vector[p] + change_matrix[p, t]

    return new_markings_np, enabled_transitions_np


@cython.boundscheck(False)
@cython.wraparound(False)
def _bfs_core_cython(
    long[:] initial_marking,
    int[:, :] pre_matrix,
    int[:, :] change_matrix,
    long place_upper_limit,
    long max_markings_to_explore
):
    """Core BFS loop optimized with Cython."""
    cdef long marking_index_counter = 0
    cdef Py_ssize_t num_places = initial_marking.shape[0]
    cdef Py_ssize_t num_transitions = pre_matrix.shape[1]

    # Store visited markings as a vector of vectors or just list of arrays
    cdef list visited_markings_list = []

    cdef np.ndarray[np.int64_t, ndim=1] initial_marking_np = np.empty(num_places, dtype=np.int64)
    cdef Py_ssize_t i, j, p, t
    for p in range(num_places):
        initial_marking_np[p] = initial_marking[p]
    visited_markings_list.append(initial_marking_np)

    cdef unordered_map[unsigned long long, long] explored_markings_dict
    explored_markings_dict[fnv1a_hash(initial_marking)] = marking_index_counter

    # Queue for BFS
    cdef np.ndarray[np.int64_t, ndim=1] queue_np = np.empty(max_markings_to_explore, dtype=np.int64)
    cdef long[:] queue = queue_np
    queue[0] = marking_index_counter

    cdef long head = 0
    cdef long tail = 1

    cdef list reachability_edges = []
    cdef list edge_transition_indices = []
    cdef bint is_bounded = True

    cdef long current_marking_index
    cdef np.ndarray[np.int64_t, ndim=1] current_marking_np
    cdef long[:] current_marking

    cdef tuple enabled_res
    cdef np.ndarray[np.int64_t, ndim=2] enabled_next_markings
    cdef np.ndarray[np.int64_t, ndim=1] enabled_transition_indices
    cdef long[:, :] enabled_next_markings_view
    cdef long[:] enabled_transition_indices_view

    cdef bint exceeds_limit
    cdef np.ndarray[np.int64_t, ndim=1] new_marking_np
    cdef unsigned long long new_marking_hash
    cdef long existing_index

    while head < tail:
        current_marking_index = queue[head]
        head += 1

        current_marking_np = visited_markings_list[current_marking_index]
        current_marking = current_marking_np

        if len(visited_markings_list) >= max_markings_to_explore:
            is_bounded = False
            break

        enabled_res = get_enabled_transitions_cython(pre_matrix, change_matrix, current_marking)
        enabled_next_markings = enabled_res[0]
        enabled_transition_indices = enabled_res[1]

        if enabled_next_markings.size > 0:
            enabled_next_markings_view = enabled_next_markings
            enabled_transition_indices_view = enabled_transition_indices
            exceeds_limit = False

            for i in range(enabled_next_markings.shape[0]):
                for j in range(num_places):
                    if enabled_next_markings_view[i, j] > place_upper_limit:
                        exceeds_limit = True
                        break
                if exceeds_limit:
                    break

            if exceeds_limit:
                is_bounded = False
                break

            for i in range(enabled_next_markings.shape[0]):
                new_marking_np = enabled_next_markings[i]
                new_marking_hash = fnv1a_hash(new_marking_np)

                if explored_markings_dict.find(new_marking_hash) == explored_markings_dict.end():
                    marking_index_counter += 1
                    if marking_index_counter >= max_markings_to_explore:
                        reachability_edges.append((current_marking_index, marking_index_counter))
                        edge_transition_indices.append(enabled_transition_indices_view[i])
                        is_bounded = False
                        break

                    visited_markings_list.append(new_marking_np)
                    explored_markings_dict[new_marking_hash] = marking_index_counter
                    queue[tail] = marking_index_counter
                    tail += 1
                    reachability_edges.append((current_marking_index, marking_index_counter))
                else:
                    existing_index = explored_markings_dict[new_marking_hash]
                    reachability_edges.append((current_marking_index, existing_index))

                edge_transition_indices.append(enabled_transition_indices_view[i])

        if not is_bounded:
            break

    return visited_markings_list, reachability_edges, edge_transition_indices, is_bounded
