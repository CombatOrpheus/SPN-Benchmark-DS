import numpy as np
from spn_datasets.generator.SPN import compute_qualitative_properties
import time

vertices = [np.random.randint(0, 5, size=(50,)).astype(np.int32) for _ in range(500)]
edges = [[i, (i + 1) % 500] for i in range(500)]

start = time.time()
for _ in range(1000):
    compute_qualitative_properties(vertices, edges)
print((time.time() - start) / 1000)
