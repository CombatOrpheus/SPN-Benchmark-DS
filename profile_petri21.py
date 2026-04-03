import time
import numpy as np
from spn_datasets.generator.PetriGenerate import add_missing_connections, delete_excess_edges

petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
# Zero out some to trigger the missing connections
petri_matrix[:, 5:10] = 0
petri_matrix[20:25, :] = 0
petri_matrix[:, -1] = 0

from numba import jit


@jit(nopython=True, cache=True)
def delete_excess_edges_fast2(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]
    num_cols = petri_matrix.shape[1]

    # Pre-allocate array for indices
    edge_indices = np.empty(max(num_places, num_cols - 1), dtype=np.int64)

    for i in range(num_places):
        count = 0
        for j in range(num_cols - 1):
            if petri_matrix[i, j] == 1:
                edge_indices[count] = j
                count += 1

        if count >= 3:
            # We want to keep 2 edges. We shuffle and keep the first 2, and zero out the rest.
            active_edges = edge_indices[:count]
            np.random.shuffle(active_edges)
            for j in range(2, count):
                petri_matrix[i, active_edges[j]] = 0

    for i in range(2 * num_transitions):
        count = 0
        for j in range(num_places):
            if petri_matrix[j, i] == 1:
                edge_indices[count] = j
                count += 1

        if count >= 3:
            active_edges = edge_indices[:count]
            np.random.shuffle(active_edges)
            for j in range(2, count):
                petri_matrix[active_edges[j], i] = 0

    return petri_matrix


# warmup
m = petri_matrix.copy()
delete_excess_edges_fast2(m, 50)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    delete_excess_edges_fast2(m, 50)
print("delete_excess_edges_fast2:", time.time() - start)
