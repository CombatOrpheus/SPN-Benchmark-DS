import numpy as np
import time
from random import choice


def generate_random_petri_net_fast(num_places, num_transitions):
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype=np.int32)
    places = np.random.permutation(num_places)
    transitions = np.random.permutation(num_transitions)

    active_places = [places[-1]]
    active_transitions = [transitions[-1]]
    p_idx = active_places[0]
    t_idx = active_transitions[0]

    if np.random.rand() <= 0.5:
        petri_matrix[p_idx, t_idx] = 1
    else:
        petri_matrix[p_idx, t_idx + num_transitions] = 1

    remaining_nodes = [(0, p) for p in places[:-1]] + [(1, t) for t in transitions[:-1]]
    np.random.shuffle(remaining_nodes)

    for node_type, idx in remaining_nodes:
        if node_type == 0:
            t_target = choice(active_transitions)
            if np.random.rand() <= 0.5:
                petri_matrix[idx, t_target] = 1
            else:
                petri_matrix[idx, t_target + num_transitions] = 1
            active_places.append(idx)
        else:
            p_target = choice(active_places)
            if np.random.rand() <= 0.5:
                petri_matrix[p_target, idx] = 1
            else:
                petri_matrix[p_target, idx + num_transitions] = 1
            active_transitions.append(idx)

    random_place = np.random.randint(0, num_places)
    petri_matrix[random_place, -1] = 1

    return petri_matrix


start = time.time()
for _ in range(5000):
    generate_random_petri_net_fast(50, 50)
print("Time:", time.time() - start)
