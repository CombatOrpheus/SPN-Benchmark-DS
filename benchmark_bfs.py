import numpy as np
import time
from src.spn_datasets.generator.ArrivableGraph import generate_reachability_graph

# Generate a synthetic petri net
num_places = 20
num_transitions = 20
np.random.seed(42)

pre_matrix = np.random.randint(0, 2, size=(num_places, num_transitions))
post_matrix = np.random.randint(0, 2, size=(num_places, num_transitions))
initial_marking = np.random.randint(0, 3, size=(num_places, 1))

incidence_matrix = np.hstack([pre_matrix, post_matrix, initial_marking])

# warm up
generate_reachability_graph(incidence_matrix, 10, 500)

start = time.time()
for _ in range(100):
    generate_reachability_graph(incidence_matrix, 10, 500)
end = time.time()
print(f"Time: {end - start:.4f}s")
