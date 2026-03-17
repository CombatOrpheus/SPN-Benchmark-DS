# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t, int64_t
from libcpp cimport bool

np.import_array()

cdef inline uint64_t fnv1a_hash(const int64_t[:] data) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef int i
    for i in range(data.shape[0]):
        h ^= <uint64_t>data[i]
        h *= 1099511628211ULL
    return h


def _bfs_core(
    np.ndarray[np.int64_t, ndim=1] initial_marking,
    np.ndarray[np.int32_t, ndim=2] pre_matrix,
    np.ndarray[np.int32_t, ndim=2] change_matrix,
    int place_upper_limit,
    int max_markings_to_explore
):
    cdef int num_places = pre_matrix.shape[0]
    cdef int num_transitions = pre_matrix.shape[1]

    cdef int marking_index_counter = 0

    cdef vector[vector[int64_t]] visited_markings_list
    cdef vector[int64_t] init_m
    cdef int p, t, i, j
    for p in range(num_places):
        init_m.push_back(initial_marking[p])
    visited_markings_list.push_back(init_m)

    cdef unordered_map[uint64_t, int64_t] explored_markings_dict
    explored_markings_dict[fnv1a_hash(initial_marking)] = marking_index_counter

    cdef vector[int64_t] queue
    queue.push_back(marking_index_counter)
    cdef int head = 0
    cdef int tail = 1

    cdef vector[pair[int64_t, int64_t]] reachability_edges
    cdef vector[int64_t] edge_transition_indices
    cdef bool is_bounded = True
    cdef bool enabled

    cdef vector[vector[int64_t]] enabled_next_markings
    cdef vector[int64_t] enabled_transition_indices
    cdef vector[int64_t] new_marking
    cdef uint64_t new_marking_hash
    cdef int64_t current_marking_index, existing_index

    while head < tail:
        current_marking_index = queue[head]
        head += 1

        if visited_markings_list.size() >= <size_t>max_markings_to_explore:
            is_bounded = False
            break

        enabled_next_markings.clear()
        enabled_transition_indices.clear()

        # get_enabled_transitions inline
        for t in range(num_transitions):
            enabled = True
            for p in range(num_places):
                if visited_markings_list[current_marking_index][p] < pre_matrix[p, t]:
                    enabled = False
                    break

            if enabled:
                new_marking.clear()
                for p in range(num_places):
                    new_marking.push_back(visited_markings_list[current_marking_index][p] + change_matrix[p, t])
                enabled_next_markings.push_back(new_marking)
                enabled_transition_indices.push_back(t)

        if not enabled_next_markings.empty():
            for i in range(enabled_next_markings.size()):
                for p in range(num_places):
                    if enabled_next_markings[i][p] > place_upper_limit:
                        is_bounded = False
                        break
                if not is_bounded:
                    break
        if not is_bounded:
            break

        for i in range(enabled_next_markings.size()):
            new_marking = enabled_next_markings[i]

            # Create a numpy array to pass to fnv1a_hash
            new_marking_hash = 14695981039346656037ULL
            for p in range(num_places):
                new_marking_hash ^= <uint64_t>new_marking[p]
                new_marking_hash *= 1099511628211ULL

            if explored_markings_dict.find(new_marking_hash) == explored_markings_dict.end():
                marking_index_counter += 1
                if marking_index_counter >= max_markings_to_explore:
                    reachability_edges.push_back(pair[int64_t, int64_t](current_marking_index, marking_index_counter))
                    edge_transition_indices.push_back(enabled_transition_indices[i])
                    is_bounded = False
                    break

                visited_markings_list.push_back(new_marking)
                explored_markings_dict[new_marking_hash] = marking_index_counter
                queue.push_back(marking_index_counter)
                tail += 1
                reachability_edges.push_back(pair[int64_t, int64_t](current_marking_index, marking_index_counter))
            else:
                existing_index = explored_markings_dict[new_marking_hash]
                reachability_edges.push_back(pair[int64_t, int64_t](current_marking_index, existing_index))

            edge_transition_indices.push_back(enabled_transition_indices[i])

        if not is_bounded:
            break

    py_visited_markings_list = []
    for i in range(visited_markings_list.size()):
        arr = np.empty(num_places, dtype=np.int64)
        for p in range(num_places):
            arr[p] = visited_markings_list[i][p]
        py_visited_markings_list.append(arr)

    py_reachability_edges = []
    for i in range(reachability_edges.size()):
        py_reachability_edges.append([reachability_edges[i].first, reachability_edges[i].second])

    py_edge_transition_indices = []
    for i in range(edge_transition_indices.size()):
        py_edge_transition_indices.append(edge_transition_indices[i])

    return py_visited_markings_list, py_reachability_edges, py_edge_transition_indices, is_bounded
