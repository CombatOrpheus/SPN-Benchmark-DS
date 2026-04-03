import numpy as np
from spn_datasets.generator.SPN import filter_spn
import time
import cProfile

matrices = [np.random.randint(0, 2, size=(10, 11)).astype(np.int32) for _ in range(100)]
for m in matrices:
    filter_spn(m)

cProfile.run("for m in matrices: filter_spn(m)", sort="cumtime")
