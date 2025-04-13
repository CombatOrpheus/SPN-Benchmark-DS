#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPN.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""

from typing import List, Tuple, Dict, Any

import numpy as np
from numpy.linalg import solve

from DataGenerate import ArrivableGraph as ArrGra


def compute_state_equation(vertices: List[Tuple[int, ...]], edges: List[Tuple[int, int]], arc_transitions: List[int],
                           lambda_values: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    Calculates the state equation for the Stochastic Petri Net.

    Args:
        vertices (List[Tuple[int, ...]]): List of reachable markings (states).
        edges (List[Tuple[int, int]]): List of edges representing transitions between states.
        arc_transitions (List[int]): List of transition indices corresponding to each edge.
        lambda_values (np.ndarray): Array of firing rates for each transition.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing the state matrix and the target vector.
    """
    number_of_vertices = len(vertices)
    augmented_state_matrix = np.zeros((number_of_vertices + 1, number_of_vertices), dtype=int)
    target_vector = np.zeros(number_of_vertices, dtype=int)
    target_vector[-1] = 1
    augmented_state_matrix[-1, :] = 1

    for current_edge, transition_index in zip(edges, arc_transitions):
        source_vertex_index, destination_vertex_index = current_edge
        augmented_state_matrix[source_vertex_index, source_vertex_index] -= lambda_values[transition_index]
        augmented_state_matrix[destination_vertex_index, source_vertex_index] += lambda_values[transition_index]

    equation_matrix = []
    for i in range(number_of_vertices - 1):
        equation_matrix.append(augmented_state_matrix[i])
    equation_matrix.append(augmented_state_matrix[-1, :])

    return equation_matrix, target_vector


# Suggested name: calculate_average_mark_numbers
def compute_average_mark_numbers(vertices: List[Tuple[int, ...]], steady_state_probabilities: np.ndarray) -> Tuple[
    np.ndarray, List[float]]:
    """
    Calculates the average number of marks (tokens) for each place in the Petri Net.

    Args:
        vertices (List[Tuple[int, ...]]): List of reachable markings (states).
        steady_state_probabilities (np.ndarray): Array of steady-state probabilities for each state.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing the mark density matrix and the list of average mark numbers for each place.
    """
    all_markings = np.array(vertices)
    distinct_tokens = np.unique(all_markings)
    num_places = all_markings.shape[1]
    num_tokens = len(distinct_tokens)

    prob_vector = steady_state_probabilities[:, np.newaxis]  # Reshape to column vector

    # Create a boolean tensor: (num_states, num_places, num_tokens)
    token_presence = (all_markings[:, :, np.newaxis] == distinct_tokens)

    # Multiply with probabilities and sum along the states axis
    marking_density_matrix = np.sum(token_presence * prob_vector[:, np.newaxis, :], axis=0)

    # Calculate the average mark numbers for each vertex (place)
    average_tokens_per_place = np.sum(marking_density_matrix * distinct_tokens, axis=1).tolist()

    return marking_density_matrix, average_tokens_per_place


def generate_stochastic_graphical_net_task(vertices_list: List[Tuple[int, ...]], edges_list: List[Tuple[int, int]],
                                           arc_transition_indices: List[int], number_of_transitions: int) -> Tuple[
    Any, Any, Any, np.ndarray]:
    """
    Generates a Stochastic Graphical Net (SGN) task by calculating steady-state probabilities and average mark numbers.

    Args:
        vertices_list (List[Tuple[int, ...]]): List of reachable markings (states).
        edges_list (List[Tuple[int, int]]): List of edges representing transitions between states.
        arc_transition_indices (List[int]): List of transition indices corresponding to each edge.
        number_of_transitions (int): The total number of transitions in the Petri Net.

    Returns:
        Tuple[Any, Any, Any, np.ndarray]: A tuple containing steady-state probabilities, mark density list, average mark numbers, and transition rates (lambda).
    """
    transition_rates = np.random.randint(1, 11, size=number_of_transitions)
    state_matrix, target_vector = compute_state_equation(vertices_list, edges_list, arc_transition_indices, transition_rates)
    steady_state_probabilities = None
    try:
        steady_state_probabilities = solve(state_matrix, target_vector.T)
        marking_density_list, average_markings_per_place = compute_average_mark_numbers(vertices_list, steady_state_probabilities)
    except np.linalg.linalg.LinAlgError:
        marking_density_list, average_markings_per_place = None, None
    return steady_state_probabilities, marking_density_list, average_markings_per_place, transition_rates


# Suggested name: convert_to_int_list
def convert_data(numpy_array_data: np.ndarray) -> List[List[int]]:
    """
    Converts a NumPy array to a list of lists of integers.

    Args:
        numpy_array_data (np.ndarray): The NumPy array to convert.

    Returns:
        List[List[int]]: The converted list of lists of integers.
    """
    return np.array(numpy_array_data).astype(int).tolist()


# Suggested name: check_if_graph_is_connected
def is_connected_graph(petri_net_matrix: np.ndarray) -> bool:
    """
    Checks if the Petri Net graph has no isolated places or transitions.

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.

    Returns:
        bool: True if no isolated places or transitions are found, False otherwise.
    """
    petri_net_matrix = np.array(petri_net_matrix)
    number_of_transitions = len(petri_net_matrix[0]) // 2

    # Vectorized check for places with no outgoing transitions
    if np.any(np.sum(petri_net_matrix[:, :-1], axis=1) == 0):
        return False

    # Vectorized check for transitions with no incoming or outgoing arcs
    if np.any(np.sum(petri_net_matrix[:, :number_of_transitions], axis=0) + np.sum(
            petri_net_matrix[:, number_of_transitions:-1], axis=0) == 0):
        return False
    return True


# Suggested name: filter_stochastic_petri_net
def filter_spn(
        petri_net_matrix: np.ndarray, place_upper_bound: int = 10, marks_lower_limit: int = 4,
        marks_upper_limit: int = 500
) -> Tuple[Dict[str, Any], bool]:
    """
    Filters a Stochastic Petri Net based on reachability and other criteria.

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.
        place_upper_bound (int): Upper bound for the number of places. Defaults to 10.
        marks_lower_limit (int): Lower limit for the number of reachable markings. Defaults to 4.
        marks_upper_limit (int): Upper limit for the number of reachable markings. Defaults to 500.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing the results dictionary and a boolean indicating success.
    """
    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.get_arr_gra(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )
    spn_results: Dict[str, Any] = {}
    if not is_bounded or len(vertices) < marks_lower_limit:
        return spn_results, False
    steady_state_probabilities, marking_densities, average_markings, transition_rates = generate_stochastic_graphical_net_task(
        vertices, edges, arc_transitions, num_transitions
    )
    if steady_state_probabilities is None:
        return spn_results, False

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False

    spn_results["petri_net"] = convert_data(petri_net_matrix)
    spn_results["arr_vlist"] = convert_data(vertices)
    spn_results["arr_edge"] = convert_data(edges)
    spn_results["arr_tranidx"] = convert_data(arc_transitions)
    spn_results["spn_labda"] = transition_rates.tolist()
    spn_results["spn_steadypro"] = steady_state_probabilities.tolist()
    spn_results["spn_markdens"] = marking_densities.tolist()
    spn_results["spn_allmus"] = np.array(average_markings).tolist()
    spn_results["spn_mu"] = np.sum(average_markings)

    return spn_results, True


def generate_stochastic_graphical_net_task_with_given_rates(vertices_list: List[Tuple[int, ...]], edges_list: List[Tuple[int, int]],
                                                            arc_transition_indices: List[int], transition_rates: np.ndarray) -> Tuple[
    Any, Any, Any]:
    """
    Generates a Stochastic Graphical Net (SGN) task with given transition rates.

    Args:
        vertices_list (List[Tuple[int, ...]]): List of reachable markings (states).
        edges_list (List[Tuple[int, int]]): List of edges representing transitions between states.
        arc_transition_indices (List[int]): List of transition indices corresponding to each edge.
        transition_rates (np.ndarray): Array of firing rates for each transition.

    Returns:
        Tuple[Any, Any, Any]: A tuple containing steady-state probabilities, mark density list, and average mark numbers.
    """
    state_matrix, target_vector = compute_state_equation(vertices_list, edges_list, arc_transition_indices, transition_rates)
    steady_state_probabilities = None
    try:
        steady_state_probabilities = solve(state_matrix, target_vector.T)
        marking_density_list, average_markings_per_place = compute_average_mark_numbers(vertices_list, steady_state_probabilities)
    except np.linalg.linalg.LinAlgError:
        marking_density_list, average_markings_per_place = None, None
    return steady_state_probabilities, marking_density_list, average_markings_per_place


def get_stochastic_petri_net(petri_net_matrix: np.ndarray, vertices: List[Tuple[int, ...]], edges: List[Tuple[int, int]],
                             arc_transitions: List[int], transition_rates: np.ndarray) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves information about a Stochastic Petri Net given its structure and transition rates.

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.
        vertices (List[Tuple[int, ...]]): List of reachable markings (states).
        edges (List[Tuple[int, int]]): List of edges representing transitions between states.
        arc_transitions (List[int]): List of transition indices corresponding to each edge.
        transition_rates (np.ndarray): Array of firing rates for each transition.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing the results dictionary and a boolean indicating success.
    """
    spn_results: Dict[str, Any] = {}
    steady_state_probabilities, marking_densities, average_markings = generate_stochastic_graphical_net_task_with_given_rates(
        vertices, edges, arc_transitions, transition_rates
    )
    if steady_state_probabilities is None:
        return spn_results, False
    if not is_connected_graph(petri_net_matrix):
        return spn_results, False
    spn_results["petri_net"] = convert_data(petri_net_matrix)
    spn_results["arr_vlist"] = convert_data(vertices)
    spn_results["arr_edge"] = convert_data(edges)
    spn_results["arr_tranidx"] = convert_data(arc_transitions)
    spn_results["spn_labda"] = transition_rates.tolist()
    spn_results["spn_steadypro"] = steady_state_probabilities.tolist()
    spn_results["spn_markdens"] = marking_densities.tolist()
    spn_results["spn_allmus"] = np.array(average_markings).tolist()
    spn_results["spn_mu"] = np.sum(average_markings)
    return spn_results, True


# Suggested name: get_stochastic_petri_net_data_set_3
def get_spnds3(
        petri_net_matrix: np.ndarray,
        transition_rates: np.ndarray,
        place_upper_bound: int = 10,
        marks_lower_limit: int = 4,
        marks_upper_limit: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves information about a Stochastic Petri Net for a specific dataset configuration.

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.
        transition_rates (np.ndarray): Array of firing rates for each transition.
        place_upper_bound (int): Upper bound for the number of places. Defaults to 10.
        marks_lower_limit (int): Lower limit for the number of reachable markings. Defaults to 4.
        marks_upper_limit (int): Upper limit for the number of reachable markings. Defaults to 500.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing the results dictionary and a boolean indicating success.
    """
    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.get_arr_gra(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )
    spn_results: Dict[str, Any] = {}
    if not is_bounded or len(vertices) < marks_lower_limit:
        return spn_results, False
    steady_state_probabilities, marking_densities, average_markings = generate_stochastic_graphical_net_task_with_given_rates(
        vertices, edges, arc_transitions, transition_rates
    )
    if steady_state_probabilities is None:
        return spn_results, False

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False
    sum_of_average_markings = np.sum(average_markings)
    if sum_of_average_markings < -100 and sum_of_average_markings > 100:
        return spn_results, False
    spn_results["petri_net"] = convert_data(petri_net_matrix)
    spn_results["arr_vlist"] = convert_data(vertices)
    spn_results["arr_edge"] = convert_data(edges)
    spn_results["arr_tranidx"] = convert_data(arc_transitions)
    spn_results["spn_labda"] = transition_rates.tolist()
    spn_results["spn_steadypro"] = steady_state_probabilities.tolist()
    spn_results["spn_markdens"] = marking_densities.tolist()
    spn_results["spn_allmus"] = np.array(average_markings).tolist()
    spn_results["spn_mu"] = np.sum(average_markings)
    return spn_results, True
