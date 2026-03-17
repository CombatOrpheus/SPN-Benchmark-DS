# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool

np.import_array()

def _generate_candidate_matrices(
    np.ndarray[np.int32_t, ndim=2] base_petri_matrix,
    bool enable_delete_edge,
    bool enable_add_edge,
    bool enable_add_token,
    bool enable_delete_token
):
    cdef int num_places = base_petri_matrix.shape[0]
    cdef int num_cols = base_petri_matrix.shape[1]
    cdef int r, c, i, j

    # We cannot use vector[np.ndarray] easily, so we just use a Python list.
    candidate_matrices = []

    # Delete an edge
    if enable_delete_edge:
        for r in range(num_places):
            for c in range(num_cols - 1):
                if base_petri_matrix[r, c] == 1:
                    modified_matrix = base_petri_matrix.copy()
                    modified_matrix[r, c] = 0
                    candidate_matrices.append(modified_matrix)

    # Add an edge
    if enable_add_edge:
        for r in range(num_places):
            for c in range(num_cols - 1):
                if base_petri_matrix[r, c] == 0:
                    modified_matrix = base_petri_matrix.copy()
                    modified_matrix[r, c] = 1
                    candidate_matrices.append(modified_matrix)

    # Add a token
    if enable_add_token:
        for r in range(num_places):
            modified_matrix = base_petri_matrix.copy()
            modified_matrix[r, num_cols - 1] += 1
            candidate_matrices.append(modified_matrix)

    # Delete a token
    cdef int total_tokens = 0
    for r in range(num_places):
        total_tokens += base_petri_matrix[r, num_cols - 1]

    if enable_delete_token and total_tokens > 1:
        for r in range(num_places):
            if base_petri_matrix[r, num_cols - 1] >= 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, num_cols - 1] -= 1
                candidate_matrices.append(modified_matrix)

    return candidate_matrices
