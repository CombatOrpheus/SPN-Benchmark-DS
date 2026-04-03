import numpy as np
from spn_datasets.generator.PetriGenerate import generate_random_petri_net
from spn_datasets.generator.SPN import filter_spn
import time

matrices = []
for _ in range(100):
    matrices.append(generate_random_petri_net(10, 5))

start = time.time()
for m in matrices:
    filter_spn(m)
print("Time filter_spn:", time.time() - start)

start = time.time()
for _ in range(100):
    generate_random_petri_net(10, 5)
print("Time generate:", time.time() - start)
