import numpy as np
import time
from spn_datasets.generator.SPN import filter_spn

petri_matrix = np.random.randint(0, 2, size=(10, 21)).astype(np.int32)
petri_matrix[0, 0] = 1

# Warmup
filter_spn(petri_matrix)

import cProfile

cProfile.run("for _ in range(100): filter_spn(petri_matrix)", sort="cumtime")
