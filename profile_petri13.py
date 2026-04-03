import time
import numpy as np
from spn_datasets.generator.PetriGenerate import generate_random_petri_net

start = time.time()
for _ in range(20000):
    generate_random_petri_net(10, 5)
print("generate:", time.time() - start)
