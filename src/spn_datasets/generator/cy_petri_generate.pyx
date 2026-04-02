# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def delete_excess_edges_cython(
    int[:, :] petri_matrix,
    int num_transitions
):
    """Deletes excess edges from the Petri net."""
    cdef Py_ssize_t num_places = petri_matrix.shape[0]
    cdef Py_ssize_t num_cols = petri_matrix.shape[1]

    cdef np.ndarray[np.int32_t, ndim=2] new_matrix = np.empty((num_places, num_cols), dtype=np.int32)
    cdef Py_ssize_t r, c
    for r in range(num_places):
        for c in range(num_cols):
            new_matrix[r, c] = petri_matrix[r, c]

    cdef int[:, :] new_matrix_view = new_matrix
    cdef int sum_edges
    cdef list edge_indices
    cdef Py_ssize_t i, j, to_remove

    for i in range(num_places):  # Iterate over places
        sum_edges = 0
        edge_indices = []
        for j in range(num_cols - 1):
            if new_matrix_view[i, j] == 1:
                sum_edges += 1
                edge_indices.append(j)

        if sum_edges >= 3:
            # We shuffle in Python logic since using random from C is complex
            np.random.shuffle(edge_indices)
            for j in range(sum_edges - 2):
                new_matrix_view[i, edge_indices[j]] = 0

    for i in range(2 * num_transitions):  # Iterate over transitions
        sum_edges = 0
        edge_indices = []
        for j in range(num_places):
            if new_matrix_view[j, i] == 1:
                sum_edges += 1
                edge_indices.append(j)

        if sum_edges >= 3:
            np.random.shuffle(edge_indices)
            for j in range(sum_edges - 2):
                new_matrix_view[edge_indices[j], i] = 0

    return new_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
def add_missing_connections_cython(
    int[:, :] petri_matrix,
    int num_transitions
):
    """Adds connections to ensure the Petri net is valid."""
    cdef Py_ssize_t num_places = petri_matrix.shape[0]
    cdef Py_ssize_t num_cols = petri_matrix.shape[1]

    cdef np.ndarray[np.int32_t, ndim=2] new_matrix = np.empty((num_places, num_cols), dtype=np.int32)
    cdef Py_ssize_t r, c
    for r in range(num_places):
        for c in range(num_cols):
            new_matrix[r, c] = petri_matrix[r, c]

    cdef int[:, :] new_matrix_view = new_matrix
    cdef int sum_col
    cdef Py_ssize_t i, j

    # Ensure each transition has at least one connection
    for j in range(2 * num_transitions):
        sum_col = 0
        for i in range(num_places):
            sum_col += new_matrix_view[i, j]
        if sum_col == 0:
            new_matrix_view[np.random.randint(0, num_places), j] = 1

    # Ensure each place has at least one incoming and one outgoing edge
    cdef int sum_pre, sum_post
    for i in range(num_places):
        sum_pre = 0
        for j in range(num_transitions):
            sum_pre += new_matrix_view[i, j]
        if sum_pre == 0:
            new_matrix_view[i, np.random.randint(0, num_transitions)] = 1

        sum_post = 0
        for j in range(num_transitions, 2 * num_transitions):
            sum_post += new_matrix_view[i, j]
        if sum_post == 0:
            new_matrix_view[i, np.random.randint(0, num_transitions) + num_transitions] = 1

    return new_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
def add_tokens_randomly_cython(
    int[:, :] petri_matrix
):
    """Adds tokens to random places in the Petri net."""
    cdef Py_ssize_t num_places = petri_matrix.shape[0]
    cdef Py_ssize_t num_cols = petri_matrix.shape[1]

    cdef np.ndarray[np.int32_t, ndim=2] new_matrix = np.empty((num_places, num_cols), dtype=np.int32)
    cdef Py_ssize_t r, c
    for r in range(num_places):
        for c in range(num_cols):
            new_matrix[r, c] = petri_matrix[r, c]

    cdef int[:, :] new_matrix_view = new_matrix

    cdef np.ndarray[np.int64_t, ndim=1] random_values = np.random.randint(0, 10, size=num_places)
    cdef Py_ssize_t i
    for i in range(num_places):
        if random_values[i] <= 2:
            new_matrix_view[i, num_cols - 1] += 1

    return new_matrix


def prune_petri_net_cython(int[:, :] petri_matrix):
    """Prunes a Petri net by deleting edges and adding nodes."""
    cdef int num_transitions = (petri_matrix.shape[1] - 1) // 2
    cdef np.ndarray[np.int32_t, ndim=2] mat1 = delete_excess_edges_cython(petri_matrix, num_transitions)
    cdef np.ndarray[np.int32_t, ndim=2] mat2 = add_missing_connections_cython(mat1, num_transitions)
    return mat2
