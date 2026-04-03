import time
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def delete_excess_edges_fast(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]
    num_cols = petri_matrix.shape[1]
    num_edges = num_cols - 1

    for i in range(num_places):
        # Calculate sum manually instead of np.sum to avoid intermediate array allocation
        row_sum = 0
        for j in range(num_edges):
            row_sum += petri_matrix[i, j]

        if row_sum >= 3:
            edge_indices = np.empty(num_edges, dtype=np.int64)
            count = 0
            for j in range(num_edges):
                if petri_matrix[i, j] == 1:
                    edge_indices[count] = j
                    count += 1
            if count > 2:
                edge_indices = edge_indices[:count]
                indices_to_remove = np.random.permutation(edge_indices)[: count - 2]
                for j in range(len(indices_to_remove)):
                    petri_matrix[i, indices_to_remove[j]] = 0

    for i in range(2 * num_transitions):
        col_sum = 0
        for j in range(num_places):
            col_sum += petri_matrix[j, i]

        if col_sum >= 3:
            edge_indices = np.empty(num_places, dtype=np.int64)
            count = 0
            for j in range(num_places):
                if petri_matrix[j, i] == 1:
                    edge_indices[count] = j
                    count += 1
            if count > 2:
                edge_indices = edge_indices[:count]
                indices_to_remove = np.random.permutation(edge_indices)[: count - 2]
                for j in range(len(indices_to_remove)):
                    petri_matrix[indices_to_remove[j], i] = 0

    return petri_matrix


petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
# warmup
m = petri_matrix.copy()
delete_excess_edges_fast(m, 50)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    delete_excess_edges_fast(m, 50)
print("delete_excess_edges_fast:", time.time() - start)
