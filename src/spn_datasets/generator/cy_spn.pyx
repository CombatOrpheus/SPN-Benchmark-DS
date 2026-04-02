# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_state_equation_cython(
    int num_vertices,
    int[:, :] edges,
    int[:] arc_transitions,
    double[:] lambda_values
):
    """Cython-optimized core of compute_state_equation."""
    cdef Py_ssize_t num_edges = edges.shape[0]
    cdef Py_ssize_t num_entries = 2 * num_edges + num_vertices

    cdef np.ndarray[np.int32_t, ndim=1] rows = np.zeros(num_entries, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] cols = np.zeros(num_entries, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.zeros(num_entries, dtype=np.float64)

    cdef int[:] rows_view = rows
    cdef int[:] cols_view = cols
    cdef double[:] data_view = data

    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i
    cdef int src_idx, dest_idx, trans_idx
    cdef double rate

    for i in range(num_edges):
        src_idx = edges[i, 0]
        dest_idx = edges[i, 1]
        trans_idx = arc_transitions[i]
        rate = lambda_values[trans_idx]

        rows_view[idx] = src_idx
        cols_view[idx] = src_idx
        data_view[idx] = -rate
        idx += 1

        rows_view[idx] = dest_idx
        cols_view[idx] = src_idx
        data_view[idx] = rate
        idx += 1

    for i in range(num_vertices):
        rows_view[idx] = num_vertices
        cols_view[idx] = i
        data_view[idx] = 1.0
        idx += 1

    return data, rows, cols


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_average_markings_cython(
    long[:, :] vertices,
    double[:] steady_state_probs
):
    """Calculates the average number of tokens for each place."""
    cdef Py_ssize_t num_states = vertices.shape[0]
    cdef Py_ssize_t num_places = vertices.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] avg_tokens_per_place = np.zeros(num_places, dtype=np.float64)
    cdef double[:] avg_tokens_view = avg_tokens_per_place

    cdef Py_ssize_t s, p
    cdef long max_token = 0
    cdef long val
    cdef double prob

    # Calculate max token to pre-allocate tracking arrays
    for s in range(num_states):
        for p in range(num_places):
            if vertices[s, p] > max_token:
                max_token = vertices[s, p]

    cdef np.ndarray[np.uint8_t, ndim=1] present_tokens = np.zeros(max_token + 1, dtype=np.uint8)
    cdef unsigned char[:] present_tokens_view = present_tokens

    for s in range(num_states):
        prob = steady_state_probs[s]
        for p in range(num_places):
            val = vertices[s, p]
            avg_tokens_view[p] += val * prob
            present_tokens_view[val] = 1

    cdef long num_unique_tokens = 0
    cdef long i
    for i in range(max_token + 1):
        if present_tokens_view[i]:
            num_unique_tokens += 1

    cdef np.ndarray[np.int32_t, ndim=1] token_to_idx = np.full(max_token + 1, -1, dtype=np.int32)
    cdef int[:] token_to_idx_view = token_to_idx

    cdef int idx = 0
    for i in range(max_token + 1):
        if present_tokens_view[i]:
            token_to_idx_view[i] = idx
            idx += 1

    cdef np.ndarray[np.float64_t, ndim=2] marking_density_matrix = np.zeros((num_places, num_unique_tokens), dtype=np.float64)
    cdef double[:, :] marking_density_view = marking_density_matrix

    cdef long token_val
    for s in range(num_states):
        prob = steady_state_probs[s]
        for p in range(num_places):
            token_val = vertices[s, p]
            idx = token_to_idx_view[token_val]
            marking_density_view[p, idx] += prob

    return marking_density_matrix, avg_tokens_per_place


@cython.boundscheck(False)
@cython.wraparound(False)
def is_connected_cython(int[:, :] petri_net_matrix):
    """Checks if the Petri net is weakly connected."""
    cdef Py_ssize_t num_places = petri_net_matrix.shape[0]
    cdef Py_ssize_t num_cols = petri_net_matrix.shape[1]

    if num_places == 0 or num_cols < 3:
        return False

    cdef Py_ssize_t num_transitions = (num_cols - 1) // 2
    if num_transitions == 0:
        return False

    cdef Py_ssize_t p, c, t, u, v
    cdef bint has_edge

    # Check for isolated places
    for p in range(num_places):
        has_edge = False
        for c in range(2 * num_transitions):
            if petri_net_matrix[p, c] != 0:
                has_edge = True
                break
        if not has_edge:
            return False

    # Check for isolated transitions
    for t in range(num_transitions):
        has_edge = False
        for p in range(num_places):
            if petri_net_matrix[p, t] != 0 or petri_net_matrix[p, num_transitions + t] != 0:
                has_edge = True
                break
        if not has_edge:
            return False

    # BFS
    cdef Py_ssize_t num_nodes = num_places + num_transitions
    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(num_nodes, dtype=np.uint8)
    cdef unsigned char[:] visited_view = visited

    cdef np.ndarray[np.int32_t, ndim=1] queue = np.empty(num_nodes, dtype=np.int32)
    cdef int[:] queue_view = queue

    cdef int head = 0
    cdef int tail = 0

    queue_view[tail] = 0
    tail += 1
    visited_view[0] = 1
    cdef int count = 0

    while head < tail:
        u = queue_view[head]
        head += 1
        count += 1

        if u < num_places:
            p = u
            for t in range(num_transitions):
                if petri_net_matrix[p, t] == 1 or petri_net_matrix[p, num_transitions + t] == 1:
                    v = num_places + t
                    if not visited_view[v]:
                        visited_view[v] = 1
                        queue_view[tail] = v
                        tail += 1
        else:
            t = u - num_places
            for p in range(num_places):
                if petri_net_matrix[p, t] == 1 or petri_net_matrix[p, num_transitions + t] == 1:
                    v = p
                    if not visited_view[v]:
                        visited_view[v] = 1
                        queue_view[tail] = v
                        tail += 1

    return count == num_nodes
