#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : ArrivableGraph.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述: Generates the reachability graph for a given Petri net definition
          using Breadth-First Search (BFS) and optimized marking lookup.
"""

from collections import deque  # Import deque for efficient BFS queue

import numpy as np


def enabled_sets(pre_condition_matrix, change_matrix, current_marking_vector):
    """
    Identifies enabled transitions for the current marking and calculates
    the resulting markings if those transitions fire.

    Args:
        pre_condition_matrix: The pre-condition matrix (input arcs).
        change_matrix: The change matrix (Post - Pre).
        current_marking_vector: The current marking (state) of the Petri net.

    Returns:
        A tuple containing:
            - new_markings: Markings resulting from firing enabled transitions (NxM numpy array).
            - enabled_transitions: Indices of the enabled transitions (1D numpy array).
    """
    # Ensure current_marking_vector is correctly shaped for broadcasting
    current_marking_expanded = current_marking_vector[:, np.newaxis] # More idiomatic than np.expand_dims

    # Find indices of transitions where the current marking satisfies the pre-conditions
    enabled_transitions = np.where(np.all(current_marking_expanded >= pre_condition_matrix, axis=0))[0]

    if not enabled_transitions.size: # Check if the array is empty
        # Return empty arrays if no transitions are enabled
        num_places = pre_condition_matrix.shape[0]
        return np.empty((0, num_places), dtype=int), np.empty((0,), dtype=int)

    # Calculate the markings that result from firing each enabled transition
    # NewMarking = CurrentMarking + (PostMatrix_enabled - PreMatrix_enabled)
    #            = CurrentMarking + ChangeMatrix_enabled
    new_markings_calc = current_marking_expanded + change_matrix[:, enabled_transitions]

    # Transpose new_markings to have markings as rows (consistent with v_list)
    return new_markings_calc.T, enabled_transitions


def get_arr_gra(incidence_matrix_with_initial, place_upper_limit=10, max_markings_to_explore=500):
    """
    Obtain the reachability graph of the petri net using BFS.

    Args:
        incidence_matrix_with_initial: Petri net definition including pre-conditions,
                                       post-conditions, and initial marking. Assumes
                                       format [pre | post | M0].
        place_upper_limit: The upper bound for tokens in any single place.
                           Exceeding this suggests unboundedness.
        max_markings_to_explore: The maximum number of markings to explore before
                                 assuming unboundedness.

    Returns:
        A tuple containing:
        - visited_markings_list: The list of unique reachable markings (states).
        - reachability_edges: List of edges [from_marking_idx, to_marking_idx].
        - edge_transition_indices: List of transition indices corresponding to each edge.
        - num_transitions: Number of transitions in the Petri net.
        - bound_flag: Boolean indicating if the net appears bounded (True) or
                      unbounded (False) based on the limits.
    """

    incidence_matrix = np.array(incidence_matrix_with_initial)
    is_bounded = False  # Assume unbounded initially
    num_transitions = incidence_matrix.shape[1] // 2
    pre_matrix = incidence_matrix[:, 0:num_transitions]
    post_matrix = incidence_matrix[:, num_transitions:-1]
    initial_marking = np.array(incidence_matrix[:, -1], dtype=int)

    # Pre-calculate the change matrix (Post - Pre)
    change_matrix = post_matrix - pre_matrix

    marking_index_counter = 0
    visited_markings_list = [initial_marking]  # List to store unique marking arrays
    # Dictionary maps marking tuples to their index in visited_markings_list for O(1) lookup
    explored_markings_dict = {tuple(initial_marking): marking_index_counter}

    # Queue for BFS, storing indices of markings to explore
    processing_queue = deque([marking_index_counter])

    reachability_edges = []  # List to store edges (from_idx, to_idx)
    edge_transition_indices = []  # List to store the transition fired for each edge

    while processing_queue:  # Continue while there are markings to explore
        # Get the next marking index from the front of the queue (BFS)
        current_marking_index = processing_queue.popleft()
        current_marking = visited_markings_list[current_marking_index]

        # Check for unboundedness based on explored states limit *before* processing
        # Note: Place upper limit check happens after generating next markings
        if marking_index_counter >= max_markings_to_explore:
            processing_queue.clear()
            break

        # Find transitions enabled by the current marking and the resulting states
        enabled_next_markings, enabled_transition_indices_fired = enabled_sets(pre_matrix, change_matrix, current_marking)

        # Check for unboundedness based on token count in *next* potential markings
        if enabled_next_markings.size > 0 and np.any(enabled_next_markings > place_upper_limit):
            processing_queue.clear()
            break

        if len(enabled_next_markings) == 0:
            # No transitions enabled from this marking, continue to next in queue
            continue

        # Process each enabled transition and the resulting marking
        for next_enabled_marking, enabled_transition_index in zip(enabled_next_markings,
                                                                  enabled_transition_indices_fired):
            # Convert the resulting marking (numpy array) to a tuple for dict key
            new_marking_tuple = tuple(next_enabled_marking)

            # Check if this new marking has been visited using the dictionary (fast lookup)
            if new_marking_tuple not in explored_markings_dict:
                # New marking found
                marking_index_counter += 1

                # Check explored count limit again *before* adding
                if marking_index_counter >= max_markings_to_explore:
                    is_bounded = False
                    processing_queue.clear()  # Stop exploration
                    # Add the edge leading to this discovery, then break inner loop
                    reachability_edges.append(
                        [current_marking_index, marking_index_counter])  # Edge leads "out of bounds" conceptually
                    edge_transition_indices.append(enabled_transition_index)
                    break  # Exit the inner for loop

                # Store the new marking (array) and its index (dict)
                visited_markings_list.append(next_enabled_marking)
                explored_markings_dict[new_marking_tuple] = marking_index_counter
                # Add the *new* index to the queue for later exploration
                processing_queue.append(marking_index_counter)

                new_marking_index_to_use = marking_index_counter  # Use the new index for the edge
                reachability_edges.append([current_marking_index, new_marking_index_to_use])

            else:
                # Marking already exists, get its index from the dictionary
                new_marking_existing_index = explored_markings_dict[new_marking_tuple]
                # Add edge from current marking to the existing marking
                reachability_edges.append([current_marking_index, new_marking_existing_index])

            # Record the transition that generated this edge/marking
            edge_transition_indices.append(enabled_transition_index)

        # Check again if the inner loop was broken due to limit
        if not is_bounded and not processing_queue:
            break  # Exit the outer while loop as well

    # Loop finished (either fully explored or hit limit)
    is_bounded = True
    return visited_markings_list, reachability_edges, edge_transition_indices, num_transitions, is_bounded
