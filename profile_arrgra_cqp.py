import numpy as np
import time
import spn_datasets.generator.ArrivableGraph as ArrGra

petri_matrix = np.random.randint(0, 2, size=(10, 21)).astype(np.int32)
# Ensure connected to actually generate graph
petri_matrix[0, 0] = 1

# Warmup
ArrGra.generate_reachability_graph(petri_matrix)

start = time.time()
for _ in range(100):
    ArrGra.generate_reachability_graph(petri_matrix)
print("generate_reachability_graph:", time.time() - start)
