# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef inline unsigned long long fnv1a_hash(const long long[:] data, int n):
    """FNV-1a hash function for an array."""
    cdef unsigned long long h = 14695981039346656037ULL
    cdef int i
    for i in range(n):
        h ^= <unsigned long long>data[i]
        h *= 1099511628211ULL
    return h

def generate_reachability_graph(incidence_matrix_with_initial, int place_upper_limit=10, int max_markings_to_explore=500):
    cdef np.ndarray incidence_matrix = np.array(incidence_matrix_with_initial, dtype=np.int64)
    cdef int num_cols = incidence_matrix.shape[1]
    cdef int num_transitions = num_cols // 2
    cdef int num_places = incidence_matrix.shape[0]

    cdef np.ndarray pre_matrix = np.ascontiguousarray(incidence_matrix[:, :num_transitions], dtype=np.int64)
    cdef np.ndarray post_matrix = np.ascontiguousarray(incidence_matrix[:, num_transitions:-1], dtype=np.int64)
    cdef np.ndarray change_matrix = np.ascontiguousarray(post_matrix - pre_matrix, dtype=np.int64)
    cdef np.ndarray initial_marking = np.ascontiguousarray(incidence_matrix[:, -1], dtype=np.int64)

    # Convert arrays to memoryviews for fast C-level access
    cdef long long[:, :] pre_view = pre_matrix
    cdef long long[:, :] change_view = change_matrix
    cdef long long[:] m0_view = initial_marking

    # Result containers
    cdef vector[vector[long long]] visited_markings
    cdef vector[vector[int]] reachability_edges
    cdef vector[int] edge_transition_indices
    cdef unordered_map[unsigned long long, int] explored_markings

    cdef bint is_bounded = True
    cdef int marking_index_counter = 0

    cdef vector[long long] m0_vec
    cdef int i, p, t
    for i in range(num_places):
        m0_vec.push_back(m0_view[i])

    visited_markings.push_back(m0_vec)
    explored_markings[fnv1a_hash(m0_view, num_places)] = marking_index_counter

    # Queue for BFS
    cdef vector[int] queue
    queue.push_back(marking_index_counter)
    cdef int head = 0

    cdef int current_marking_index
    cdef vector[long long] current_marking
    cdef vector[long long] new_marking
    cdef unsigned long long new_marking_hash
    cdef bint enabled
    cdef bint valid_next

    # Pre-allocate array for fast hash calculation
    cdef long long* new_marking_arr = <long long*>malloc(num_places * sizeof(long long))

    while head < queue.size():
        current_marking_index = queue[head]
        head += 1
        current_marking = visited_markings[current_marking_index]

        if visited_markings.size() >= max_markings_to_explore:
            is_bounded = False
            break

        for t in range(num_transitions):
            enabled = True
            for p in range(num_places):
                if current_marking[p] < pre_view[p, t]:
                    enabled = False
                    break

            if enabled:
                new_marking.clear()
                valid_next = True
                for p in range(num_places):
                    new_val = current_marking[p] + change_view[p, t]
                    if new_val > place_upper_limit:
                        valid_next = False
                        is_bounded = False
                        break
                    new_marking.push_back(new_val)
                    new_marking_arr[p] = new_val

                if not valid_next:
                    break

                new_marking_hash = fnv1a_hash(<long long[:num_places]>new_marking_arr, num_places)

                if explored_markings.find(new_marking_hash) == explored_markings.end():
                    marking_index_counter += 1
                    if marking_index_counter >= max_markings_to_explore:
                        reachability_edges.push_back([current_marking_index, marking_index_counter])
                        edge_transition_indices.push_back(t)
                        is_bounded = False
                        break

                    visited_markings.push_back(new_marking)
                    explored_markings[new_marking_hash] = marking_index_counter
                    queue.push_back(marking_index_counter)
                    reachability_edges.push_back([current_marking_index, marking_index_counter])
                else:
                    existing_index = explored_markings[new_marking_hash]
                    reachability_edges.push_back([current_marking_index, existing_index])

                edge_transition_indices.push_back(t)

        if not is_bounded:
            break

    free(new_marking_arr)

    # Convert C++ vectors back to Python lists/numpy arrays
    py_visited_markings = []
    for i in range(visited_markings.size()):
        arr = np.zeros(num_places, dtype=np.int64)
        for p in range(num_places):
            arr[p] = visited_markings[i][p]
        py_visited_markings.append(arr)

    py_reachability_edges = []
    for i in range(reachability_edges.size()):
        py_reachability_edges.append([reachability_edges[i][0], reachability_edges[i][1]])

    py_edge_transition_indices = []
    for i in range(edge_transition_indices.size()):
        py_edge_transition_indices.append(edge_transition_indices[i])

    return (
        py_visited_markings,
        py_reachability_edges,
        py_edge_transition_indices,
        num_transitions,
        is_bounded,
    )
