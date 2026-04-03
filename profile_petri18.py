import time
import numpy as np
from spn_datasets.generator.PetriGenerate import delete_excess_edges

petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
# warmup
m = petri_matrix.copy()
delete_excess_edges(m, 50)

start = time.time()
for _ in range(10000):
    m = petri_matrix.copy()
    delete_excess_edges(m, 50)
print("delete_excess_edges:", time.time() - start)
