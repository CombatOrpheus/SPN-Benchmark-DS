import time
import numpy as np
from spn_datasets.generator.PetriGenerate import add_missing_connections, delete_excess_edges

petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
# Zero out some to trigger the missing connections
petri_matrix[:, 5:10] = 0
petri_matrix[20:25, :] = 0

# warmup
m = petri_matrix.copy()
add_missing_connections(m, 50)
m = petri_matrix.copy()
delete_excess_edges(m, 50)


def test_delete_excess_edges():
    for _ in range(10000):
        m = petri_matrix.copy()
        delete_excess_edges(m, 50)


import cProfile

cProfile.run("test_delete_excess_edges()", sort="cumtime")
