"""
This module provides functions for generating and modifying Petri nets.
"""

from random import choice
import numpy as np
import numba


def generate_random_petri_net(num_places, num_transitions):
    """Generates a random Petri net matrix.

    Args:
        num_places (int): The number of places.
        num_transitions (int): The number of transitions.

    Returns:
        np.ndarray: The generated Petri net matrix.
    """
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype=np.int32)

    places = list(range(num_places))
    transitions = list(range(num_transitions))

    # Shuffle to ensure randomness
    np.random.shuffle(places)
    np.random.shuffle(transitions)

    # Initialize with one place and one transition
    active_places = [places.pop()]
    active_transitions = [transitions.pop()]

    # Connect the first pair
    p_idx = active_places[0]
    t_idx = active_transitions[0]

    if np.random.rand() <= 0.5:
        petri_matrix[p_idx, t_idx] = 1  # P -> T
    else:
        petri_matrix[p_idx, t_idx + num_transitions] = 1  # T -> P

    # Collect remaining nodes
    remaining_nodes = []
    for p in places:
        remaining_nodes.append(("P", p))
    for t in transitions:
        remaining_nodes.append(("T", t))

    np.random.shuffle(remaining_nodes)

    # Connect remaining nodes to the existing graph
    for node_type, idx in remaining_nodes:
        if node_type == "P":
            # Connect new place to an existing transition
            t_target = choice(active_transitions)
            if np.random.rand() <= 0.5:
                petri_matrix[idx, t_target] = 1  # P -> T
            else:
                petri_matrix[idx, t_target + num_transitions] = 1  # T -> P
            active_places.append(idx)
        else:
            # Connect new transition to an existing place
            p_target = choice(active_places)
            if np.random.rand() <= 0.5:
                petri_matrix[p_target, idx] = 1  # P -> T
            else:
                petri_matrix[p_target, idx + num_transitions] = 1  # T -> P
            active_transitions.append(idx)

    # Add an initial marking to a random place
    random_place = np.random.randint(0, num_places)
    petri_matrix[random_place, -1] = 1

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def prune_petri_net(petri_matrix):
    """Prunes a Petri net by deleting edges and adding nodes.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.

    Returns:
        np.ndarray: The pruned Petri net matrix.
    """
    num_transitions = (petri_matrix.shape[1] - 1) // 2
    petri_matrix = delete_excess_edges(petri_matrix, num_transitions)
    petri_matrix = add_missing_connections(petri_matrix, num_transitions)
    return petri_matrix


@numba.jit(nopython=True, cache=True)
def delete_excess_edges(petri_matrix, num_transitions):
    """Deletes excess edges from the Petri net.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.
        num_transitions (int): The number of transitions.

    Returns:
        np.ndarray: The matrix with excess edges removed.
    """
    for i in range(petri_matrix.shape[0]):  # Iterate over places
        if np.sum(petri_matrix[i, :-1]) >= 3:
            edge_indices = np.empty(petri_matrix.shape[1] - 1, dtype=np.int64)
            count = 0
            for j in range(petri_matrix.shape[1] - 1):
                if petri_matrix[i, j] == 1:
                    edge_indices[count] = j
                    count += 1
            if count > 2:
                edge_indices = edge_indices[:count]
                indices_to_remove = np.random.permutation(edge_indices)[: count - 2]
                for j in range(len(indices_to_remove)):
                    petri_matrix[i, indices_to_remove[j]] = 0

    for i in range(2 * num_transitions):  # Iterate over transitions
        if np.sum(petri_matrix[:, i]) >= 3:
            edge_indices = np.empty(petri_matrix.shape[0], dtype=np.int64)
            count = 0
            for j in range(petri_matrix.shape[0]):
                if petri_matrix[j, i] == 1:
                    edge_indices[count] = j
                    count += 1
            if count > 2:
                edge_indices = edge_indices[:count]
                indices_to_remove = np.random.permutation(edge_indices)[: count - 2]
                for j in range(len(indices_to_remove)):
                    petri_matrix[indices_to_remove[j], i] = 0

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def add_missing_connections(petri_matrix, num_transitions):
    """Adds connections to ensure the Petri net is valid.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.
        num_transitions (int): The number of transitions.

    Returns:
        np.ndarray: The matrix with missing connections added.
    """
    # Ensure each transition has at least one connection
    pre_matrix = petri_matrix[:, :num_transitions]
    post_matrix = petri_matrix[:, num_transitions:-1]

    # Pre-allocate tracking arrays to avoid np.where inside Numba
    zero_sum_cols = np.empty(2 * num_transitions, dtype=np.int64)
    zero_sum_cols_count = 0
    col_sums = np.sum(petri_matrix[:, : 2 * num_transitions], axis=0)
    for i in range(len(col_sums)):
        if col_sums[i] == 0:
            zero_sum_cols[zero_sum_cols_count] = i
            zero_sum_cols_count += 1

    if zero_sum_cols_count > 0:
        zero_sum_cols = zero_sum_cols[:zero_sum_cols_count]
        random_rows = np.random.randint(0, petri_matrix.shape[0], size=zero_sum_cols_count)
        for i in range(zero_sum_cols_count):
            petri_matrix[random_rows[i], zero_sum_cols[i]] = 1

    # Ensure each place has at least one incoming and one outgoing edge
    rows_with_zero_pre_sum = np.empty(petri_matrix.shape[0], dtype=np.int64)
    rows_with_zero_pre_sum_count = 0
    pre_row_sums = np.sum(pre_matrix, axis=1)
    for i in range(len(pre_row_sums)):
        if pre_row_sums[i] == 0:
            rows_with_zero_pre_sum[rows_with_zero_pre_sum_count] = i
            rows_with_zero_pre_sum_count += 1

    if rows_with_zero_pre_sum_count > 0:
        rows_with_zero_pre_sum = rows_with_zero_pre_sum[:rows_with_zero_pre_sum_count]
        random_cols_pre = np.random.randint(0, num_transitions, size=rows_with_zero_pre_sum_count)
        for i in range(rows_with_zero_pre_sum_count):
            petri_matrix[rows_with_zero_pre_sum[i], random_cols_pre[i]] = 1

    rows_with_zero_post_sum = np.empty(petri_matrix.shape[0], dtype=np.int64)
    rows_with_zero_post_sum_count = 0
    post_row_sums = np.sum(post_matrix, axis=1)
    for i in range(len(post_row_sums)):
        if post_row_sums[i] == 0:
            rows_with_zero_post_sum[rows_with_zero_post_sum_count] = i
            rows_with_zero_post_sum_count += 1

    if rows_with_zero_post_sum_count > 0:
        rows_with_zero_post_sum = rows_with_zero_post_sum[:rows_with_zero_post_sum_count]
        random_cols_post = np.random.randint(0, num_transitions, size=rows_with_zero_post_sum_count)
        for i in range(rows_with_zero_post_sum_count):
            petri_matrix[rows_with_zero_post_sum[i], random_cols_post[i] + num_transitions] = 1

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def add_tokens_randomly(petri_matrix):
    """Adds tokens to random places in the Petri net.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.

    Returns:
        np.ndarray: The matrix with tokens added.
    """
    random_values = np.random.randint(0, 10, size=petri_matrix.shape[0])
    petri_matrix[:, -1] += (random_values <= 2).astype(np.int32)
    return petri_matrix
