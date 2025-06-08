#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPN.py
# @Date    : 2020-08-22 (modified for optimized NumPy and sparse matrix construction)
# @Author  : mingjian
    描述
"""

from typing import List, Tuple, Dict, Any  # Python's built-in typing

import numpy as np
from scipy.sparse.linalg import lsmr, lsqr
from scipy.sparse import csc_array, lil_matrix  # Added lil_matrix import

from DataGenerate import ArrivableGraph as ArrGra


def compute_state_equation(vertices: List[np.ndarray], edges: List[List[int]], arc_transitions: List[int],
                           lambda_values: np.ndarray) -> Tuple[csc_array, np.ndarray]:
    """
    Calculates the state equation for the Stochastic Petri Net using sparse matrices.

    Args:
        vertices (List[np.ndarray]): List of reachable markings (states), each marking is a 1D NumPy array.
        edges (List[List[int]]): List of edges representing transitions between states.
        arc_transitions (List[int]): List of transition indices corresponding to each edge.
        lambda_values (np.ndarray): Array of firing rates for each transition.

    Returns:
        Tuple[csc_array, np.ndarray]: Sparse matrix (CSC format) for the system of equations
                                     and the target vector.
    """
    number_of_vertices = len(vertices)

    # Initialize as a LIL (List of Lists) sparse matrix for efficient construction
    augmented_state_matrix_sparse = lil_matrix((number_of_vertices + 1, number_of_vertices), dtype=float)

    # Fill in the transition rate effects
    for current_edge, transition_index in zip(edges, arc_transitions):
        source_vertex_index, destination_vertex_index = current_edge
        rate = lambda_values[transition_index]
        augmented_state_matrix_sparse[source_vertex_index, source_vertex_index] -= rate
        augmented_state_matrix_sparse[destination_vertex_index, source_vertex_index] += rate

    # Constraint: sum of steady-state probabilities must be 1
    # lil_matrix supports setting a whole row like this
    augmented_state_matrix_sparse[number_of_vertices, :] = 1.0

    target_vector = np.zeros(number_of_vertices + 1, dtype=float)
    target_vector[number_of_vertices] = 1.0

    # Convert to CSC (Compressed Sparse Column) format for efficient arithmetic operations by solvers
    equation_matrix_csc = augmented_state_matrix_sparse.tocsc()

    return equation_matrix_csc, target_vector


def compute_average_mark_numbers(vertices_np: np.ndarray, steady_state_probabilities: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Calculates the average number of marks (tokens) for each place in the Petri Net.

    Args:
        vertices_np (np.ndarray): 2D NumPy array of reachable markings (states).
        steady_state_probabilities (np.ndarray): Array of steady-state probabilities for each state.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mark density matrix
                                       and the NumPy array of average mark numbers for each place.
    """
    prob_vector_col = steady_state_probabilities.reshape(-1, 1)
    average_tokens_per_place = np.sum(vertices_np * prob_vector_col, axis=0)

    distinct_token_values = np.unique(vertices_np)
    num_places = vertices_np.shape[1]
    num_distinct_tokens = len(distinct_token_values)

    marking_density_matrix = np.zeros((num_places, num_distinct_tokens), dtype=float)

    for place_idx in range(num_places):
        for token_val_idx, token_val in enumerate(distinct_token_values):
            states_with_token_val = (vertices_np[:, place_idx] == token_val)
            marking_density_matrix[place_idx, token_val_idx] = np.sum(steady_state_probabilities[states_with_token_val])

    return marking_density_matrix, average_tokens_per_place


def generate_stochastic_graphical_net_task(
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transition_indices: List[int],
        number_of_transitions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Generates an SGN task: calculates steady-state probabilities and average mark numbers.

    Args:
        vertices_list (List[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (List[List[int]]): List of edges.
        arc_transition_indices (List[int]): Transition indices for each edge.
        number_of_transitions (int): Total number of transitions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        Steady-state probabilities, mark density, average mark numbers, lambda rates, and success flag.
        Returns None for array types if unsuccessful.
    """
    transition_rates = np.random.randint(1, 11, size=number_of_transitions).astype(float)

    if not vertices_list:
        return None, None, None, transition_rates, False
    vertices_np = np.array(vertices_list, dtype=int)

    state_matrix_sparse, target_vector_eq = compute_state_equation(
        vertices_list, edges_list, arc_transition_indices, transition_rates
    )

    steady_state_probabilities = None
    marking_density_array = None
    average_markings_array = None
    success = False

    try:
        lsmr_result = lsmr(state_matrix_sparse, target_vector_eq, atol=1e-7, btol=1e-7, conlim=1e7, maxiter=None)

        if lsmr_result[1] in [1, 2]:
            steady_state_probabilities = lsmr_result[0]
            steady_state_probabilities[steady_state_probabilities < 0] = 0
            if np.sum(steady_state_probabilities) > 1e-9:
                steady_state_probabilities /= np.sum(steady_state_probabilities)
            else:
                return None, None, None, transition_rates, False

            marking_density_array, average_markings_array = compute_average_mark_numbers(
                vertices_np, steady_state_probabilities
            )
            success = True
        else:
            pass

    except np.linalg.LinAlgError as e:
        pass
    except ValueError as ve:
        pass

    return steady_state_probabilities, marking_density_array, average_markings_array, transition_rates, success


def is_connected_graph(petri_net_matrix: np.ndarray) -> bool:
    """
    Checks if the Petri Net graph has no isolated places or transitions.
    Assumes petri_net_matrix is (num_places, 2*num_transitions + 1).

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.

    Returns:
        bool: True if no isolated places or transitions are found, False otherwise.
    """
    if petri_net_matrix.size == 0: return False
    num_places = petri_net_matrix.shape[0]
    if num_places == 0: return False

    num_cols = petri_net_matrix.shape[1]
    if num_cols < 3: return False

    num_transitions = (num_cols - 1) // 2
    if num_transitions == 0: return False

    if np.any(np.sum(petri_net_matrix[:, :2 * num_transitions], axis=1) == 0):
        return False

    pre_matrix_sum_cols = np.sum(petri_net_matrix[:, :num_transitions], axis=0)
    post_matrix_sum_cols = np.sum(petri_net_matrix[:, num_transitions:2 * num_transitions], axis=0)
    if np.any(pre_matrix_sum_cols + post_matrix_sum_cols == 0):
        return False

    return True


def filter_stochastic_petri_net(
        petri_net_matrix: np.ndarray, place_upper_bound: int = 10, marks_lower_limit: int = 4,
        marks_upper_limit: int = 500
) -> Tuple[Dict[str, Any], bool]:
    """
    Filters a Stochastic Petri Net based on reachability and other criteria.
    Returns dictionary with NumPy arrays for relevant fields.

    Args:
        petri_net_matrix (np.ndarray): The Petri Net matrix.
        place_upper_bound (int): Upper bound for tokens in any single place.
        marks_lower_limit (int): Lower limit for the number of reachable markings.
        marks_upper_limit (int): Upper limit for the number of reachable markings.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing the results dictionary
                                     (with NumPy arrays) and a boolean indicating success.
    """
    spn_results: Dict[str, Any] = {}

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False

    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.get_arr_gra(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return spn_results, False

    vertices_np = np.array(vertices, dtype=int)

    (steady_state_probs, marking_densities_arr,
     average_markings_arr, firing_rates_arr, sgn_success) = generate_stochastic_graphical_net_task(
        vertices, edges, arc_transitions, num_transitions
    )

    if not sgn_success or steady_state_probs is None:
        return spn_results, False

    current_spn_mu = np.sum(average_markings_arr)
    if not (-1000 <= current_spn_mu <= 1000):
        return spn_results, False

    spn_results["petri_net"] = petri_net_matrix
    spn_results["arr_vlist"] = vertices_np
    spn_results["arr_edge"] = np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    spn_results["arr_tranidx"] = np.array(arc_transitions, dtype=np.int32) if arc_transitions else np.empty((0,), dtype=np.int32)

    spn_results["spn_labda"] = firing_rates_arr
    spn_results["spn_steadypro"] = steady_state_probs
    spn_results["spn_markdens"] = marking_densities_arr
    spn_results["spn_allmus"] = average_markings_arr
    spn_results["spn_mu"] = current_spn_mu

    return spn_results, True


def generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transition_indices: List[int],
        transition_rates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Generates an SGN task with given transition rates.

    Args:
        vertices_list (List[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (List[List[int]]): List of edges.
        arc_transition_indices (List[int]): Transition indices for each edge.
        transition_rates (np.ndarray): Array of firing rates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        Steady-state probabilities, mark density, average mark numbers, and success flag.
        Returns None for array types if unsuccessful.
    """
    if not vertices_list:
        return None, None, None, False
    vertices_np = np.array(vertices_list, dtype=int)

    transition_rates_float = np.array(transition_rates, dtype=float)

    state_matrix_sparse, target_vector_eq = compute_state_equation(
        vertices_list, edges_list, arc_transition_indices, transition_rates_float
    )

    steady_state_probabilities = None
    marking_density_array = None
    average_markings_array = None
    success = False

    try:
        lsmr_result = lsmr(state_matrix_sparse, target_vector_eq, atol=1e-7, btol=1e-7, conlim=1e7, maxiter=None)
        if lsmr_result[1] in [1, 2]:
            steady_state_probabilities = lsmr_result[0]
            steady_state_probabilities[steady_state_probabilities < 0] = 0
            if np.sum(steady_state_probabilities) > 1e-9:
                steady_state_probabilities /= np.sum(steady_state_probabilities)
            else:
                return None, None, None, False

            marking_density_array, average_markings_array = compute_average_mark_numbers(
                vertices_np, steady_state_probabilities
            )
            success = True
        else:
            pass
    except np.linalg.LinAlgError as e:
        pass
    except ValueError as ve:
        pass

    return steady_state_probabilities, marking_density_array, average_markings_array, success


def get_stochastic_petri_net(
        petri_net_matrix_input: np.ndarray,
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transitions_list: List[int],
        transition_rates_input: np.ndarray
) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves SPN info given structure and transition rates. Returns dict with NumPy arrays.

    Args:
        petri_net_matrix_input (np.ndarray): The Petri Net matrix.
        vertices_list (List[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (List[List[int]]): List of edges.
        arc_transitions_list (List[int]): Transition indices for each edge.
        transition_rates_input (np.ndarray): Array of firing rates.

    Returns:
        Tuple[Dict[str, Any], bool]: Results dictionary and success flag.
    """
    petri_net_matrix = np.array(petri_net_matrix_input, dtype=int)
    transition_rates = np.array(transition_rates_input, dtype=float)

    spn_results: Dict[str, Any] = {}

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False

    if not vertices_list:
        return spn_results, False
    vertices_np = np.array(vertices_list, dtype=int)

    (steady_state_probs, marking_densities_arr,
     average_markings_arr, sgn_success) = generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list, edges_list, arc_transitions_list, transition_rates
    )

    if not sgn_success or steady_state_probs is None:
        return spn_results, False

    current_spn_mu = np.sum(average_markings_arr)

    spn_results["petri_net"] = petri_net_matrix
    spn_results["arr_vlist"] = vertices_np
    spn_results["arr_edge"] = np.array(edges_list, dtype=int) if edges_list else np.empty((0, 2), dtype=int)
    spn_results["arr_tranidx"] = np.array(arc_transitions_list, dtype=int) if arc_transitions_list else np.empty((0,),
                                                                                                                 dtype=int)

    spn_results["spn_labda"] = transition_rates
    spn_results["spn_steadypro"] = steady_state_probs
    spn_results["spn_markdens"] = marking_densities_arr
    spn_results["spn_allmus"] = average_markings_arr
    spn_results["spn_mu"] = current_spn_mu

    return spn_results, True


def get_spnds3(
        petri_net_matrix_input: np.ndarray,
        transition_rates_input: np.ndarray,
        place_upper_bound: int = 10,
        marks_lower_limit: int = 4,
        marks_upper_limit: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves SPN info for a specific dataset configuration (DS3 like).
    Returns dict with NumPy arrays.

    Args:
        petri_net_matrix_input (np.ndarray): Petri Net matrix.
        transition_rates_input (np.ndarray): Firing rates.
        place_upper_bound (int): Config param.
        marks_lower_limit (int): Config param.
        marks_upper_limit (int): Config param.

    Returns:
        Tuple[Dict[str, Any], bool]: Results dictionary and success flag.
    """
    petri_net_matrix = np.array(petri_net_matrix_input, dtype=int)
    transition_rates = np.array(transition_rates_input, dtype=float)
    spn_results: Dict[str, Any] = {}

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False

    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.get_arr_gra(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return spn_results, False

    vertices_np = np.array(vertices, dtype=int)

    (steady_state_probs, marking_densities_arr,
     average_markings_arr, sgn_success) = generate_stochastic_graphical_net_task_with_given_rates(
        vertices, edges, arc_transitions, transition_rates
    )

    if not sgn_success or steady_state_probs is None:
        return spn_results, False

    current_spn_mu = np.sum(average_markings_arr)
    if not (-1000 <= current_spn_mu <= 1000):
        return spn_results, False

    spn_results["petri_net"] = petri_net_matrix
    spn_results["arr_vlist"] = vertices_np
    spn_results["arr_edge"] = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)
    spn_results["arr_tranidx"] = np.array(arc_transitions, dtype=int) if arc_transitions else np.empty((0,), dtype=int)

    spn_results["spn_labda"] = transition_rates
    spn_results["spn_steadypro"] = steady_state_probs
    spn_results["spn_markdens"] = marking_densities_arr
    spn_results["spn_allmus"] = average_markings_arr
    spn_results["spn_mu"] = current_spn_mu

    return spn_results, True
