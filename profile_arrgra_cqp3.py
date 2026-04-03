import numpy as np
import time
from spn_datasets.generator.ArrivableGraph import _bfs_core

petri_matrix = np.random.randint(0, 2, size=(10, 21)).astype(np.int32)
petri_matrix[0, 0] = 1

num_transitions = 10
pre_matrix = petri_matrix[:, :num_transitions]
post_matrix = petri_matrix[:, num_transitions:-1]
initial_marking = np.array(petri_matrix[:, -1], dtype=np.int64)
change_matrix = post_matrix - pre_matrix

# Warmup
_bfs_core(initial_marking, pre_matrix, change_matrix, 10, 500)

start = time.time()
for _ in range(100):
    _bfs_core(initial_marking, pre_matrix, change_matrix, 10, 500)
print("_bfs_core:", time.time() - start)
