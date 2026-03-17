# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand
import random

np.import_array()

def delete_excess_edges(np.ndarray[np.int32_t, ndim=2] petri_matrix, int num_transitions):
    cdef int num_places = petri_matrix.shape[0]
    cdef int num_cols = petri_matrix.shape[1]
    cdef int i, j

    for i in range(num_places):
        edge_count = 0
        for j in range(num_cols - 1):
            if petri_matrix[i, j] == 1:
                edge_count += 1

        if edge_count >= 3:
            indices = []
            for j in range(num_cols - 1):
                if petri_matrix[i, j] == 1:
                    indices.append(j)

            np.random.shuffle(indices)
            for j in range(len(indices) - 2):
                petri_matrix[i, indices[j]] = 0

    for i in range(2 * num_transitions):
        edge_count = 0
        for j in range(num_places):
            if petri_matrix[j, i] == 1:
                edge_count += 1

        if edge_count >= 3:
            indices = []
            for j in range(num_places):
                if petri_matrix[j, i] == 1:
                    indices.append(j)

            np.random.shuffle(indices)
            for j in range(len(indices) - 2):
                petri_matrix[indices[j], i] = 0

    return petri_matrix


def add_missing_connections(np.ndarray[np.int32_t, ndim=2] petri_matrix, int num_transitions):
    cdef int num_places = petri_matrix.shape[0]
    cdef int i, j

    # Ensure each transition has at least one connection
    for i in range(2 * num_transitions):
        sum_col = 0
        for j in range(num_places):
            sum_col += petri_matrix[j, i]

        if sum_col == 0:
            random_row = np.random.randint(0, num_places)
            petri_matrix[random_row, i] = 1

    # Ensure each place has at least one incoming and one outgoing edge
    for i in range(num_places):
        sum_pre = 0
        for j in range(num_transitions):
            sum_pre += petri_matrix[i, j]

        if sum_pre == 0:
            random_col = np.random.randint(0, num_transitions)
            petri_matrix[i, random_col] = 1

        sum_post = 0
        for j in range(num_transitions, 2 * num_transitions):
            sum_post += petri_matrix[i, j]

        if sum_post == 0:
            random_col = np.random.randint(0, num_transitions)
            petri_matrix[i, random_col + num_transitions] = 1

    return petri_matrix


def add_tokens_randomly(np.ndarray[np.int32_t, ndim=2] petri_matrix):
    cdef int num_places = petri_matrix.shape[0]
    cdef int num_cols = petri_matrix.shape[1]
    cdef int i

    random_values = np.random.randint(0, 10, size=num_places)
    for i in range(num_places):
        if random_values[i] <= 2:
            petri_matrix[i, num_cols - 1] += 1

    return petri_matrix


def prune_petri_net(np.ndarray[np.int32_t, ndim=2] petri_matrix):
    cdef int num_transitions = (petri_matrix.shape[1] - 1) // 2
    petri_matrix = delete_excess_edges(petri_matrix, num_transitions)
    petri_matrix = add_missing_connections(petri_matrix, num_transitions)
    return petri_matrix
