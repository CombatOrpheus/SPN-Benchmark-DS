#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : ArrivableGraph.py
@Date    : 2020-08-22
@Author  : mingjian

This module provides functions to generate the reachability graph of a
Stochastic Petri Net (SPN) using a Breadth-First Search (BFS) algorithm.
"""

from collections import deque
from typing import List, Tuple, Dict, Set

import numpy as np


def get_enabled_transitions(
    pre_condition_matrix: np.ndarray,
    change_matrix: np.ndarray,
    current_marking: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies enabled transitions for a given marking and calculates the resulting markings.

    Args:
        pre_condition_matrix: The pre-condition (input) matrix of the Petri net.
        change_matrix: The change matrix (Post - Pre).
        current_marking: The current marking (state) of the Petri net.

    Returns:
        A tuple containing:
        - The new markings that result from firing each enabled transition.
        - The indices of the transitions that were enabled.
    """
    # Use broadcasting to find all transitions that are enabled by the current marking.
    # A transition is enabled if the marking has enough tokens in all its input places.
    enabled_mask = np.all(current_marking[:, np.newaxis] >= pre_condition_matrix, axis=0)
    enabled_transition_indices = np.where(enabled_mask)[0]

    if enabled_transition_indices.size == 0:
        num_places = pre_condition_matrix.shape[0]
        return np.empty((0, num_places), dtype=int), np.empty((0,), dtype=int)

    # Calculate the new markings by adding the change matrix columns for enabled transitions.
    new_markings = current_marking[:, np.newaxis] + change_matrix[:, enabled_transition_indices]

    return new_markings.T, enabled_transition_indices


def build_reachability_graph(
    petri_net_matrix: np.ndarray,
    place_upper_bound: int = 10,
    max_markings_to_explore: int = 500,
) -> Tuple[List[np.ndarray], List[List[int]], List[int], int, bool]:
    """
    Builds the reachability graph of a Petri net using Breadth-First Search (BFS).

    The exploration is bounded by the number of markings and the token count in any place
    to handle potentially unbounded nets.

    Args:
        petri_net_matrix: The matrix defining the Petri net [Pre | Post | M0].
        place_upper_bound: The max number of tokens in any place before considering the net unbounded.
        max_markings_to_explore: The max number of unique markings to explore.

    Returns:
        A tuple containing:
        - A list of all unique reachable markings (the states of the graph).
        - A list of all edges in the graph, represented as [from_index, to_index].
        - A list of the transition indices that fire for each edge.
        - The total number of transitions in the net.
        - A boolean flag indicating if the net was found to be bounded.
    """
    num_transitions = petri_net_matrix.shape[1] // 2
    pre_matrix = petri_net_matrix[:, :num_transitions]
    post_matrix = petri_net_matrix[:, num_transitions:-1]
    initial_marking = petri_net_matrix[:, -1].astype(int)
    change_matrix = post_matrix - pre_matrix

    # Data structures for BFS
    visited_markings: List[np.ndarray] = [initial_marking]
    # Use a dictionary for O(1) lookup of visited markings
    explored_markings_map: Dict[Tuple[int, ...], int] = {tuple(initial_marking): 0}
    queue = deque([0])  # Queue stores indices of markings to visit

    edges: List[List[int]] = []
    edge_transitions: List[int] = []
    is_bounded = True

    while queue:
        current_marking_idx = queue.popleft()
        current_marking = visited_markings[current_marking_idx]

        # Check for unboundedness based on the number of explored states.
        if len(visited_markings) >= max_markings_to_explore:
            is_bounded = False
            break

        # Find all enabled transitions and the markings they lead to.
        next_markings, fired_transitions = get_enabled_transitions(
            pre_matrix, change_matrix, current_marking
        )

        # Check for unboundedness based on the token count in any place.
        if next_markings.size > 0 and np.any(next_markings > place_upper_bound):
            is_bounded = False
            break

        # Process each new marking.
        for new_marking, transition_idx in zip(next_markings, fired_transitions):
            new_marking_tuple = tuple(new_marking)

            if new_marking_tuple not in explored_markings_map:
                # This is a newly discovered marking.
                new_marking_idx = len(visited_markings)
                explored_markings_map[new_marking_tuple] = new_marking_idx
                visited_markings.append(new_marking)
                queue.append(new_marking_idx)
                edges.append([current_marking_idx, new_marking_idx])
            else:
                # This marking has been seen before.
                existing_marking_idx = explored_markings_map[new_marking_tuple]
                edges.append([current_marking_idx, existing_marking_idx])

            edge_transitions.append(transition_idx)

    return visited_markings, edges, edge_transitions, num_transitions, is_bounded
