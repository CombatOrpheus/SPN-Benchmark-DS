import numpy as np
from spn_datasets.generator.ArrivableGraph import get_enabled_transitions, _bfs_core
import time


def run_prof():
    matrices = [np.random.randint(0, 2, size=(10, 21)).astype(np.int32) for _ in range(500)]
    start = time.time()
    for m in matrices:
        num_transitions = 10
        pre_matrix = m[:, :num_transitions]
        post_matrix = m[:, num_transitions:-1]
        initial_marking = np.array(m[:, -1], dtype=np.int64)
        change_matrix = post_matrix - pre_matrix
        _bfs_core(initial_marking, pre_matrix, change_matrix, 10, 500)
    print("Time:", time.time() - start)


run_prof()
