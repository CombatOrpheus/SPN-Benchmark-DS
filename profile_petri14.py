import time
import numpy as np
from spn_datasets.generator.ArrivableGraph import get_enabled_transitions

num_places = 50
num_transitions = 50
pre_condition_matrix = np.random.randint(0, 3, size=(num_places, num_transitions)).astype(np.int64)
change_matrix = np.random.randint(-1, 2, size=(num_places, num_transitions)).astype(np.int64)
current_marking_vector = np.random.randint(0, 5, size=(num_places,)).astype(np.int64)

# warmup
get_enabled_transitions(pre_condition_matrix, change_matrix, current_marking_vector)

start = time.time()
for _ in range(50000):
    get_enabled_transitions(pre_condition_matrix, change_matrix, current_marking_vector)
print("get_enabled_transitions:", time.time() - start)
