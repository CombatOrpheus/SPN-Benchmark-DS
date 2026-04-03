import numpy as np
import time
import cProfile

import spn_datasets.generator.ArrivableGraph as ArrGra

petri_matrix = np.random.randint(0, 2, size=(10, 21)).astype(np.int32)
petri_matrix[0, 0] = 1


def run_prof():
    for _ in range(100):
        ArrGra.generate_reachability_graph(petri_matrix)


cProfile.run("run_prof()", sort="cumtime")
