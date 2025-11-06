"""
This module provides functions for generating and modifying Petri nets.
"""

from random import choice
import numpy as np
import numba


def _initialize_petri_net(num_places, num_transitions):
    """Initializes the Petri net matrix and selects the first connection."""
    remaining_nodes = list(range(1, num_places + num_transitions + 1))
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype="int32")

    first_place = choice(range(num_places)) + 1
    first_transition = choice(range(num_transitions)) + num_places + 1

    remaining_nodes.remove(first_place)
    remaining_nodes.remove(first_transition)

    if np.random.rand() <= 0.5:
        petri_matrix[first_place - 1, first_transition - num_places - 1] = 1
    else:
        petri_matrix[first_place - 1, first_transition - num_places - 1 + num_transitions] = 1

    np.random.shuffle(remaining_nodes)
    sub_graph = np.array([first_place, first_transition])
    return petri_matrix, remaining_nodes, sub_graph


def _connect_remaining_nodes(petri_matrix, remaining_nodes, sub_graph, num_places, num_transitions):
    """Connects the remaining nodes to the sub-graph."""
    for node in np.random.permutation(remaining_nodes):
        sub_places = sub_graph[sub_graph <= num_places]
        sub_transitions = sub_graph[sub_graph > num_places]

        if node <= num_places:
            place = node
            transition = choice(sub_transitions)
        else:
            place = choice(sub_places)
            transition = node

        if np.random.rand() <= 0.5:
            petri_matrix[place - 1, transition - num_places - 1] = 1
        else:
            petri_matrix[place - 1, transition - num_places - 1 + num_transitions] = 1

        sub_graph = np.concatenate((sub_graph, [node]))
    return petri_matrix


def generate_random_petri_net(num_places, num_transitions):
    """Generates a random Petri net matrix.

    Args:
        num_places (int): The number of places.
        num_transitions (int): The number of transitions.

    Returns:
        np.ndarray: The generated Petri net matrix.
    """
    petri_matrix, remaining_nodes, sub_graph = _initialize_petri_net(num_places, num_transitions)
    petri_matrix = _connect_remaining_nodes(petri_matrix, remaining_nodes, sub_graph, num_places, num_transitions)

    # Add an initial marking
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
            edge_indices = np.where(petri_matrix[i, :-1] == 1)[0]
            if len(edge_indices) > 2:
                indices_to_remove = np.random.permutation(edge_indices)[: len(edge_indices) - 2]
                petri_matrix[i, indices_to_remove] = 0

    for i in range(2 * num_transitions):  # Iterate over transitions
        if np.sum(petri_matrix[:, i]) >= 3:
            edge_indices = np.where(petri_matrix[:, i] == 1)[0]
            if len(edge_indices) > 2:
                indices_to_remove = np.random.permutation(edge_indices)[: len(edge_indices) - 2]
                petri_matrix[indices_to_remove, i] = 0

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

    zero_sum_cols = np.where(np.sum(petri_matrix[:, : 2 * num_transitions], axis=0) == 0)[0]
    random_rows = np.random.randint(0, petri_matrix.shape[0], size=len(zero_sum_cols))
    for i in range(len(zero_sum_cols)):
        petri_matrix[random_rows[i], zero_sum_cols[i]] = 1

    # Ensure each place has at least one incoming and one outgoing edge
    rows_with_zero_pre_sum = np.where(np.sum(pre_matrix, axis=1) == 0)[0]
    random_cols_pre = np.random.randint(0, num_transitions, size=len(rows_with_zero_pre_sum))
    for i in range(len(rows_with_zero_pre_sum)):
        petri_matrix[rows_with_zero_pre_sum[i], random_cols_pre[i]] = 1

    rows_with_zero_post_sum = np.where(np.sum(post_matrix, axis=1) == 0)[0]
    random_cols_post = np.random.randint(0, num_transitions, size=len(rows_with_zero_post_sum))
    for i in range(len(rows_with_zero_post_sum)):
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
