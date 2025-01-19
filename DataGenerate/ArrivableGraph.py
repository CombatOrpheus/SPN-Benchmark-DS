#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : ArrivableGraph.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""

import numpy as np
import numba
from .wherevec import wherevec_cython


@numba.jit(nopython=True, cache=True)
def wherevec(vec, matrix):
    for row in range(len(matrix)):
        if np.allclose(matrix[row, :], vec):
            return row
    return -1


def enabled_sets(pre_set, post_set, markings):
    """
    Finds enabled transitions and calculates new markings.

    Args:
        pre_set: Preset matrix.
        post_set: Postset matrix.
        markings: Current markings (M).

    Returns:
        Tuple containing the array of enabled transitions and the array of new markings.
    """
    markings = np.expand_dims(markings, axis=1)  # Add dimension for broadcasting
    enabled = np.all(markings >= pre_set, axis=0)  # Check if all preconditions are met for each transition

    enabled_transitions = np.where(enabled)[0]
    new_markings = markings - pre_set[:, enabled_transitions] + post_set[:, enabled_transitions]
    return new_markings, enabled_transitions


def get_arr_gra(petri_net_matrix, place_capacity=10, max_markings=500):
    petri_net_matrix = np.array(petri_net_matrix)
    num_transitions = petri_net_matrix.shape[1] // 2

    pre_transitions = petri_net_matrix[:, 0:num_transitions]
    post_transitions = petri_net_matrix[:, num_transitions:-1]
    initial_marking = petri_net_matrix[:, -1]

    reachable_markings = {tuple(initial_marking): 0}
    unvisited_markings = {0}
    edges = []
    transitions_taken = []
    marking_change_matrix = post_transitions - pre_transitions
    marking_index_counter = 0

    if np.any(initial_marking > place_capacity):
        return list(reachable_markings.keys()), edges, transitions_taken, num_transitions, False

    while unvisited_markings:
        current_marking_index = unvisited_markings.pop()
        current_marking = np.array(list(reachable_markings.keys())[current_marking_index])

        enabled_sets_result = enabled_sets(pre_transitions, post_transitions, current_marking)

        if enabled_sets_result[0] is None:
            continue

        next_markings, enabled_transitions = enabled_sets_result

        if np.any(next_markings > place_capacity) or marking_index_counter > max_markings:
            return list(reachable_markings.keys()), edges, transitions_taken, num_transitions, False

        for next_marking, transition in zip(next_markings.T, enabled_transitions):
            next_marking_tuple = tuple(next_marking)
            if next_marking_tuple not in reachable_markings:
                marking_index_counter += 1
                reachable_markings[next_marking_tuple] = marking_index_counter
                unvisited_markings.add(marking_index_counter)
            edges.append((current_marking_index, reachable_markings[next_marking_tuple]))
            transitions_taken.append(transition)

    return list(reachable_markings.keys()), edges, transitions_taken, num_transitions, True