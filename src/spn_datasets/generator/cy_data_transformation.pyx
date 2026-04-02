# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython

from libcpp.vector cimport vector

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def _generate_candidate_matrices_cython(
    int[:, :] base_petri_matrix,
    bint enable_delete_edge,
    bint enable_add_edge,
    bint enable_add_token,
    bint enable_delete_token,
):
    """Cython-optimized function to generate candidate Petri net matrices."""
    cdef list candidate_matrices = []
    cdef Py_ssize_t num_places = base_petri_matrix.shape[0]
    cdef Py_ssize_t num_cols = base_petri_matrix.shape[1]

    cdef np.ndarray[np.int32_t, ndim=2] base_petri_matrix_np = np.empty((num_places, num_cols), dtype=np.int32)
    cdef Py_ssize_t r, c
    for r in range(num_places):
        for c in range(num_cols):
            base_petri_matrix_np[r, c] = base_petri_matrix[r, c]

    cdef np.ndarray[np.int32_t, ndim=2] modified_matrix
    cdef int[:, :] modified_matrix_view

    cdef int total_tokens = 0
    if enable_delete_token:
        for r in range(num_places):
            total_tokens += base_petri_matrix[r, num_cols - 1]

    # Delete an edge
    if enable_delete_edge:
        for r in range(num_places):
            for c in range(num_cols - 1):
                if base_petri_matrix[r, c] == 1:
                    modified_matrix = base_petri_matrix_np.copy()
                    modified_matrix_view = modified_matrix
                    modified_matrix_view[r, c] = 0
                    candidate_matrices.append(modified_matrix)

    # Add an edge
    if enable_add_edge:
        for r in range(num_places):
            for c in range(num_cols - 1):
                if base_petri_matrix[r, c] == 0:
                    modified_matrix = base_petri_matrix_np.copy()
                    modified_matrix_view = modified_matrix
                    modified_matrix_view[r, c] = 1
                    candidate_matrices.append(modified_matrix)

    # Add a token
    if enable_add_token:
        for r in range(num_places):
            modified_matrix = base_petri_matrix_np.copy()
            modified_matrix_view = modified_matrix
            modified_matrix_view[r, num_cols - 1] += 1
            candidate_matrices.append(modified_matrix)

    # Delete a token
    if enable_delete_token and total_tokens > 1:
        for r in range(num_places):
            if base_petri_matrix[r, num_cols - 1] >= 1:
                modified_matrix = base_petri_matrix_np.copy()
                modified_matrix_view = modified_matrix
                modified_matrix_view[r, num_cols - 1] -= 1
                candidate_matrices.append(modified_matrix)

    return candidate_matrices
