import numpy as np
import time
from numba import jit


@jit(nopython=True, cache=True)
def generate_random_petri_net_fast2(num_places, num_transitions):
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype=np.int32)
    places = np.random.permutation(num_places)
    transitions = np.random.permutation(num_transitions)

    active_places = np.empty(num_places, dtype=np.int32)
    active_transitions = np.empty(num_transitions, dtype=np.int32)
    ap_count = 1
    at_count = 1

    active_places[0] = places[-1]
    active_transitions[0] = transitions[-1]

    p_idx = active_places[0]
    t_idx = active_transitions[0]

    if np.random.rand() <= 0.5:
        petri_matrix[p_idx, t_idx] = 1
    else:
        petri_matrix[p_idx, t_idx + num_transitions] = 1

    remaining_nodes_types = np.empty(num_places - 1 + num_transitions - 1, dtype=np.int32)
    remaining_nodes_idxs = np.empty(num_places - 1 + num_transitions - 1, dtype=np.int32)

    for i in range(num_places - 1):
        remaining_nodes_types[i] = 0
        remaining_nodes_idxs[i] = places[i]
    for i in range(num_transitions - 1):
        remaining_nodes_types[num_places - 1 + i] = 1
        remaining_nodes_idxs[num_places - 1 + i] = transitions[i]

    perm = np.random.permutation(num_places - 1 + num_transitions - 1)

    for i in range(len(perm)):
        node_type = remaining_nodes_types[perm[i]]
        idx = remaining_nodes_idxs[perm[i]]
        if node_type == 0:
            t_target = active_transitions[np.random.randint(0, at_count)]
            if np.random.rand() <= 0.5:
                petri_matrix[idx, t_target] = 1
            else:
                petri_matrix[idx, t_target + num_transitions] = 1
            active_places[ap_count] = idx
            ap_count += 1
        else:
            p_target = active_places[np.random.randint(0, ap_count)]
            if np.random.rand() <= 0.5:
                petri_matrix[p_target, idx] = 1
            else:
                petri_matrix[p_target, idx + num_transitions] = 1
            active_transitions[at_count] = idx
            at_count += 1

    random_place = np.random.randint(0, num_places)
    petri_matrix[random_place, -1] = 1

    return petri_matrix


# warmup
generate_random_petri_net_fast2(10, 5)

start = time.time()
for _ in range(5000):
    generate_random_petri_net_fast2(50, 50)
print("Time:", time.time() - start)
