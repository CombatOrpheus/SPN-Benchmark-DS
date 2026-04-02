import numpy as np
import time
from src.spn_datasets.generator.PetriGenerate import delete_excess_edges, add_missing_connections

np.random.seed(42)

petri_matrices = []
for _ in range(1000):
    petri_matrix = np.random.randint(0, 2, size=(50, 101), dtype=np.int32)
    petri_matrices.append(petri_matrix)

# Warmup
delete_excess_edges(petri_matrices[0], 50)
add_missing_connections(petri_matrices[0], 50)

start = time.time()
for m in petri_matrices:
    delete_excess_edges(m, 50)
end = time.time()
print(f"delete_excess_edges Time: {end - start:.4f}s")

start = time.time()
for m in petri_matrices:
    add_missing_connections(m, 50)
end = time.time()
print(f"add_missing_connections Time: {end - start:.4f}s")
