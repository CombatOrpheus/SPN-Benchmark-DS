import numpy as np
import time
from spn_datasets.generator.ArrivableGraph import generate_reachability_graph

petri_matrix = np.random.randint(0, 2, size=(10, 21)).astype(np.int32)
petri_matrix[0, 0] = 1

# Warmup
generate_reachability_graph(petri_matrix)

import cProfile

cProfile.run("for _ in range(100): generate_reachability_graph(petri_matrix)", sort="cumtime")
