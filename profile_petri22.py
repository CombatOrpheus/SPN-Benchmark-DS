import time
import numpy as np
from numba import jit
from spn_datasets.generator.PetriGenerate import add_missing_connections, add_tokens_randomly

petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
petri_matrix[:, 5:10] = 0
petri_matrix[20:25, :] = 0
petri_matrix[:, -1] = 0

from numba import jit


@jit(nopython=True, cache=True)
def add_missing_connections_fast(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]

    for i in range(2 * num_transitions):
        count = 0
        for j in range(num_places):
            count += petri_matrix[j, i]
        if count == 0:
            random_row = np.random.randint(0, num_places)
            petri_matrix[random_row, i] = 1

    for i in range(num_places):
        count_pre = 0
        for j in range(num_transitions):
            count_pre += petri_matrix[i, j]
        if count_pre == 0:
            random_col = np.random.randint(0, num_transitions)
            petri_matrix[i, random_col] = 1

        count_post = 0
        for j in range(num_transitions):
            count_post += petri_matrix[i, j + num_transitions]
        if count_post == 0:
            random_col = np.random.randint(0, num_transitions)
            petri_matrix[i, random_col + num_transitions] = 1

    return petri_matrix


# warmup
m = petri_matrix.copy()
add_missing_connections_fast(m, 50)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    add_missing_connections_fast(m, 50)
print("add_missing_connections_fast:", time.time() - start)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    add_missing_connections(m, 50)
print("add_missing_connections:", time.time() - start)
