import numpy as np
from spn_datasets.generator.ArrivableGraph import generate_reachability_graph
import time
import cProfile

import spn_datasets.generator.ArrivableGraph as ArrGra

ArrGra.generate_reachability_graph(np.random.randint(0, 2, size=(2, 3)).astype(np.int32))

matrices = [np.random.randint(0, 2, size=(10, 11)).astype(np.int32) for _ in range(500)]
start = time.time()
for m in matrices:
    generate_reachability_graph(m)
print("Time:", time.time() - start)
