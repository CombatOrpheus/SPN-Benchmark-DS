import numpy as np
from spn_datasets.generator.PetriGenerate import generate_random_petri_net
import time

start = time.time()
for _ in range(5000):
    generate_random_petri_net(10, 5)
print("Time:", time.time() - start)
