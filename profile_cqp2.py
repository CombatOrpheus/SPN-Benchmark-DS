import numpy as np
import time
import numba

vertices = [np.random.randint(0, 5, size=(50,)).astype(np.int32) for _ in range(500)]

start = time.time()
for _ in range(1000):
    max_tokens = int(max(np.max(v) for v in vertices))
print("Gen:", (time.time() - start) / 1000)

start = time.time()
for _ in range(1000):
    max_tokens = int(np.max(np.array(vertices)))
print("Dense:", (time.time() - start) / 1000)
