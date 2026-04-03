import time
import numpy as np
from spn_datasets.generator.PetriGenerate import add_missing_connections, delete_excess_edges

petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)

from numba import jit


@jit(nopython=True, cache=True)
def delete_excess_edges_fast(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]
    num_cols = petri_matrix.shape[1]

    # Places to transitions (pre edges) and transitions to places (post edges) are interleaved or separated?
    # Actually, the original iterates over petri_matrix[i, :-1]
    for i in range(num_places):
        # Count ones manually
        count_ones = 0
        for j in range(num_cols - 1):
            if petri_matrix[i, j] == 1:
                count_ones += 1

        if count_ones >= 3:
            edge_indices = np.empty(count_ones, dtype=np.int64)
            c = 0
            for j in range(num_cols - 1):
                if petri_matrix[i, j] == 1:
                    edge_indices[c] = j
                    c += 1
            indices_to_remove = np.random.permutation(edge_indices)[: count_ones - 2]
            for j in range(len(indices_to_remove)):
                petri_matrix[i, indices_to_remove[j]] = 0

    for i in range(2 * num_transitions):
        count_ones = 0
        for j in range(num_places):
            if petri_matrix[j, i] == 1:
                count_ones += 1

        if count_ones >= 3:
            edge_indices = np.empty(count_ones, dtype=np.int64)
            c = 0
            for j in range(num_places):
                if petri_matrix[j, i] == 1:
                    edge_indices[c] = j
                    c += 1
            indices_to_remove = np.random.permutation(edge_indices)[: count_ones - 2]
            for j in range(len(indices_to_remove)):
                petri_matrix[indices_to_remove[j], i] = 0

    return petri_matrix


# warmup
m = petri_matrix.copy()
delete_excess_edges_fast(m, 50)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    delete_excess_edges_fast(m, 50)
print("delete_excess_edges_fast:", time.time() - start)
