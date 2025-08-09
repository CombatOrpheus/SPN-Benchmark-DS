#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : PetriGenerate.py
@Date    : 2020-08-22
@Author  : mingjian

This script provides functions to generate and manipulate the structure of Petri nets.
It includes functionalities for creating random Petri nets, pruning them by adjusting
arc weights, and ensuring structural properties like node connectivity.
"""

from random import choice
from typing import List

import numpy as np


def generate_random_petri_net(num_places: int, num_transitions: int) -> np.ndarray:
    """
    Generates a random, connected Petri net matrix using a graph-growth algorithm.

    The algorithm starts with a single place and transition and iteratively adds
    new nodes (places or transitions), connecting them to the existing graph
    to ensure the final Petri net is connected.

    Args:
        num_places: The number of places in the Petri net.
        num_transitions: The number of transitions in the Petri net.

    Returns:
        A Petri net matrix of shape (num_places, 2 * num_transitions + 1).
        The columns represent [Pre-conditions | Post-conditions | Initial Marking].
    """
    # Create a list of all nodes, places are 1 to num_places, transitions are num_places + 1 and up
    all_nodes = list(range(1, num_places + num_transitions + 1))
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype="int32")

    # 1. Start with a random place and a random transition to form the initial subgraph.
    initial_place = choice(range(1, num_places + 1))
    initial_transition = choice(range(num_places + 1, num_places + num_transitions + 1))
    connected_nodes = {initial_place, initial_transition}
    all_nodes.remove(initial_place)
    all_nodes.remove(initial_transition)

    # Create the first arc (either from place to transition or vice-versa).
    if np.random.rand() <= 0.5:
        # Arc from place to transition (pre-condition)
        petri_matrix[initial_place - 1, initial_transition - num_places - 1] = 1
    else:
        # Arc from transition to place (post-condition)
        petri_matrix[initial_place - 1, initial_transition - num_places - 1 + num_transitions] = 1

    # 2. Iteratively connect the remaining nodes to the growing subgraph.
    np.random.shuffle(all_nodes)
    for node in all_nodes:
        is_place = node <= num_places

        # Select a random node from the existing subgraph to connect to.
        # If the new node is a place, connect to a transition. If a transition, connect to a place.
        if is_place:
            target_node = choice([n for n in connected_nodes if n > num_places])
            p, t = node, target_node
        else:
            target_node = choice([n for n in connected_nodes if n <= num_places])
            p, t = target_node, node

        # Create a random arc between the new node and the selected existing node.
        if np.random.rand() <= 0.5:
            petri_matrix[p - 1, t - num_places - 1] = 1
        else:
            petri_matrix[p - 1, t - num_places - 1 + num_transitions] = 1

        connected_nodes.add(node)

    # 3. Add an initial marking to a random place.
    initial_marking_place = np.random.randint(0, num_places)
    petri_matrix[initial_marking_place, -1] = 1

    # 4. Sparsely add more arcs to increase complexity (10% probability for each potential arc).
    additional_arcs_mask = np.random.randint(1, 11, size=petri_matrix.shape) == 1
    # Ensure we only add arcs where none exist.
    zero_mask = petri_matrix == 0
    petri_matrix[additional_arcs_mask & zero_mask] = 1

    return petri_matrix


def prune_petri_net(petri_matrix: np.ndarray) -> np.ndarray:
    """
    Prunes and refines the structure of a Petri net matrix.

    This process involves two steps:
    1. Deleting excess arcs from nodes with a high degree.
    2. Adding necessary arcs to ensure all nodes are connected.

    Args:
        petri_matrix: The Petri net matrix to prune.

    Returns:
        The pruned and refined Petri net matrix.
    """
    num_transitions = (petri_matrix.shape[1] - 1) // 2
    petri_matrix = delete_excess_edges(petri_matrix, num_transitions)
    petri_matrix = add_necessary_arcs(petri_matrix, num_transitions)
    return petri_matrix


def delete_excess_edges(petri_matrix: np.ndarray, num_transitions: int) -> np.ndarray:
    """
    Reduces the number of arcs for nodes with a degree of 3 or more.

    For any place or transition with too many connections, this function
    randomly removes arcs until the degree is at most 2.

    Args:
        petri_matrix: The Petri net matrix.
        num_transitions: The number of transitions.

    Returns:
        The Petri net matrix with fewer edges on high-degree nodes.
    """
    # Prune places with high degree
    for place_idx in range(petri_matrix.shape[0]):
        degree = np.sum(petri_matrix[place_idx, 0:-1])
        if degree >= 3:
            # Find all arcs connected to this place
            arc_indices = np.where(petri_matrix[place_idx, 0:-1] == 1)[0]
            # Randomly choose arcs to remove
            indices_to_remove = np.random.choice(arc_indices, size=int(degree) - 2, replace=False)
            petri_matrix[place_idx, indices_to_remove] = 0

    # Prune transitions with high degree
    for trans_idx in range(2 * num_transitions):
        degree = np.sum(petri_matrix[:, trans_idx])
        if degree >= 3:
            # Find all arcs connected to this transition
            arc_indices = np.where(petri_matrix[:, trans_idx] == 1)[0]
            # Randomly choose arcs to remove
            indices_to_remove = np.random.choice(arc_indices, size=int(degree) - 2, replace=False)
            petri_matrix[indices_to_remove, trans_idx] = 0

    return petri_matrix


def add_necessary_arcs(petri_matrix: np.ndarray, num_transitions: int) -> np.ndarray:
    """
    Adds arcs to ensure the Petri net is well-formed and connected.

    This function enforces two properties:
    1. Every transition must have at least one incoming or outgoing arc.
    2. Every place must have at least one incoming and one outgoing arc.

    Args:
        petri_matrix: The Petri net matrix.
        num_transitions: The number of transitions.

    Returns:
        The Petri net matrix with necessary arcs added.
    """
    # Ensure every transition has at least one connection.
    io_matrix = petri_matrix[:, :2 * num_transitions]
    col_sums = np.sum(io_matrix, axis=0)
    isolated_cols = np.where(col_sums == 0)[0]
    if isolated_cols.size > 0:
        random_rows = np.random.randint(0, petri_matrix.shape[0], size=len(isolated_cols))
        petri_matrix[random_rows, isolated_cols] = 1

    # Ensure every place has at least one input arc and one output arc.
    pre_matrix = petri_matrix[:, 0:num_transitions]
    post_matrix = petri_matrix[:, num_transitions:-1]

    # Add input arcs to places that have none.
    places_no_input = np.where(np.sum(post_matrix, axis=1) == 0)[0]
    if places_no_input.size > 0:
        random_transitions = np.random.randint(0, num_transitions, size=len(places_no_input))
        petri_matrix[places_no_input, random_transitions + num_transitions] = 1

    # Add output arcs to places that have none.
    places_no_output = np.where(np.sum(pre_matrix, axis=1) == 0)[0]
    if places_no_output.size > 0:
        random_transitions = np.random.randint(0, num_transitions, size=len(places_no_output))
        petri_matrix[places_no_output, random_transitions] = 1

    return petri_matrix


def add_token_to_random_place(petri_matrix: np.ndarray) -> np.ndarray:
    """
    Adds a token to random places in the Petri net with a fixed probability.

    Args:
        petri_matrix: The Petri net matrix.

    Returns:
        The Petri net matrix with potentially more tokens in the initial marking.
    """
    # For each place, there is a 30% chance of adding a token.
    add_token_mask = np.random.randint(0, 10, size=petri_matrix.shape[0]) < 3
    petri_matrix[:, -1] += add_token_mask.astype(int)
    return petri_matrix
