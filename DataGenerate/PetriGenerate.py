"""
This module provides functions for generating and modifying Petri nets.
"""

import numpy as np
import numba

def generate_random_petri_net(num_places, num_transitions, num_spns=1):
    """
    Generates one or more random Petri net matrices using a vectorized approach.

    Args:
        num_places (int): The number of places.
        num_transitions (int): The number of transitions.
        num_spns (int): The number of SPNs to generate in parallel.

    Returns:
        np.ndarray: A 3D array of shape (num_spns, num_places, 2 * num_transitions + 1)
                    if num_spns > 1, or a 2D array if num_spns == 1.
    """
    # Initialize a 3D tensor to hold the batch of Petri nets.
    # Shape: (num_spns, num_places, 2 * num_transitions + 1)
    petri_nets = np.zeros((num_spns, num_places, 2 * num_transitions + 1), dtype=np.int32)

    # For each SPN in the batch, select a random starting place and transition.
    # These arrays will hold the indices for the initial connections.
    initial_places = np.random.randint(0, num_places, size=num_spns)
    initial_transitions = np.random.randint(0, num_transitions, size=num_spns)

    # For each SPN, decide the direction of the first edge (pre- or post-transition).
    # A value of 0 means pre (place to transition), 1 means post (transition to place).
    initial_directions = np.random.randint(0, 2, size=num_spns)

    # Create the first connection for each SPN in the batch.
    # We use advanced indexing with a batch index array to update the tensor in one go.
    batch_indices = np.arange(num_spns)
    petri_nets[batch_indices, initial_places, initial_transitions + initial_directions * num_transitions] = 1

    # Keep track of which nodes (places and transitions) are connected in each SPN.
    # These boolean masks will be updated as we add new nodes.
    # Shape: (num_spns, num_places + num_transitions)
    connected_nodes = np.zeros((num_spns, num_places + num_transitions), dtype=bool)
    connected_nodes[batch_indices, initial_places] = True
    connected_nodes[batch_indices, num_places + initial_transitions] = True

    # Create a list of all nodes for each SPN and shuffle them.
    # This determines the order in which we will connect the remaining nodes.
    nodes_to_connect = np.tile(np.arange(num_places + num_transitions), (num_spns, 1))
    np.apply_along_axis(np.random.shuffle, 1, nodes_to_connect)

    # Iterate through the shuffled nodes and connect them to the existing graph.
    for i in range(num_places + num_transitions):
        current_nodes = nodes_to_connect[:, i]

        # For each SPN, check if the current node is already connected.
        # This is done by looking up its status in the `connected_nodes` mask.
        is_connected_mask = connected_nodes[batch_indices, current_nodes]

        # We only need to connect nodes that are not yet part of the graph.
        # This mask identifies the SPNs where a new connection is needed.
        needs_connection_mask = ~is_connected_mask

        if np.any(needs_connection_mask):
            # Get the indices of the SPNs that require a new connection.
            active_batch_indices = batch_indices[needs_connection_mask]

            # Get the nodes to be added for this subset of SPNs.
            nodes_to_add = current_nodes[needs_connection_mask]

            # Separate the new nodes into places and transitions.
            is_place_mask = nodes_to_add < num_places

            # For new places, connect them to a random existing transition.
            if np.any(is_place_mask):
                place_indices = active_batch_indices[is_place_mask]
                new_places = nodes_to_add[is_place_mask]

                # Get the connected transitions for each relevant SPN.
                connected_transitions = connected_nodes[place_indices, num_places:]

                # Choose a random connected transition for each new place.
                random_trans_indices = np.array([np.random.choice(np.where(row)[0]) for row in connected_transitions])

                # Decide the edge direction (pre or post).
                directions = np.random.randint(0, 2, size=len(place_indices))

                # Add the new edges to the Petri net tensor.
                petri_nets[place_indices, new_places, random_trans_indices + directions * num_transitions] = 1

                # Mark the new places as connected.
                connected_nodes[place_indices, new_places] = True

            # For new transitions, connect them to a random existing place.
            if np.any(~is_place_mask):
                transition_indices = active_batch_indices[~is_place_mask]
                new_transitions = nodes_to_add[~is_place_mask] - num_places

                # Get the connected places for each relevant SPN.
                connected_places = connected_nodes[transition_indices, :num_places]

                # Choose a random connected place for each new transition.
                random_place_indices = np.array([np.random.choice(np.where(row)[0]) for row in connected_places])

                # Decide the edge direction.
                directions = np.random.randint(0, 2, size=len(transition_indices))

                # Add the new edges to the Petri net tensor.
                petri_nets[transition_indices, random_place_indices, new_transitions + directions * num_transitions] = 1

                # Mark the new transitions as connected.
                connected_nodes[transition_indices, num_places + new_transitions] = True

    # Add an initial marking to one random place in each SPN.
    random_places_for_marking = np.random.randint(0, num_places, size=num_spns)
    petri_nets[batch_indices, random_places_for_marking, -1] = 1

    # If only one SPN was generated, return it as a 2D array to maintain backward compatibility.
    if num_spns == 1:
        return petri_nets[0]

    return petri_nets


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