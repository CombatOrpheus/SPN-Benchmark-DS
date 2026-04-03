import numpy as np
from spn_datasets.generator.PetriGenerate import delete_excess_edges, add_missing_connections, prune_petri_net
import time


def run_prof():
    matrices = [np.random.randint(0, 2, size=(50, 101)).astype(np.int32) for _ in range(1000)]
    start = time.time()
    for m in matrices:
        prune_petri_net(m)
    print("Time:", time.time() - start)


run_prof()
