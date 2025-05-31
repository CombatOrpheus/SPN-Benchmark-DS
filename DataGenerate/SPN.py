#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPN.py
# @Date    : 2020-08-22 (modified for optimized NumPy handling)
# @Author  : mingjian
    描述
"""

from typing import Tuple, Dict, Any  # Python's built-in typing

import numpy as np
from scipy.sparse.linalg import lsmr, lsqr
from scipy.sparse import csc_array

from DataGenerate import ArrivableGraph as ArrGra


def compute_state_equation(vertices: list[np.ndarray], edges: list[list[int]], arc_transitions: list[int],
                           lambda_values: np.ndarray) -> Tuple[csc_array, np.ndarray]:  # Type hints updated
    """
    Calculates the state equation for the Stochastic Petri Net.

    Args:
        vertices (list[np.ndarray]): List of reachable markings (states), each marking is a 1D NumPy array.
        edges (list[list[int]]): List of edges representing transitions between states.
        arc_transitions (list[int]): List of transition indices corresponding to each edge.
        lambda_values (np.ndarray): Array of firing rates for each transition.

    Returns:
        Tuple[csc_array, np.ndarray]: Sparse matrix for the system of equations and the target vector.
    """
    number_of_vertices = len(vertices)
    # augmented_state_matrix: The last row is for sum(probabilities) = 1
    augmented_state_matrix = np.zeros((number_of_vertices + 1, number_of_vertices), dtype=float)  # Use float for rates

    # Fill in the transition rate effects
    for current_edge, transition_index in zip(edges, arc_transitions):
        source_vertex_index, destination_vertex_index = current_edge
        rate = lambda_values[transition_index]
        augmented_state_matrix[source_vertex_index, source_vertex_index] -= rate
        augmented_state_matrix[destination_vertex_index, source_vertex_index] += rate

    # Constraint: sum of steady-state probabilities must be 1
    augmented_state_matrix[number_of_vertices, :] = 1.0

    # Target vector for the system of equations:
    # dP/dt = Q*P = 0 for steady state (first N-1 rows effectively, or N rows for Q)
    # sum(P_i) = 1 (last row)
    target_vector = np.zeros(number_of_vertices + 1, dtype=float)
    target_vector[number_of_vertices] = 1.0  # For the sum(P_i) = 1 constraint

    # For lsqr/lsmr, we typically solve A*x = b.
    # The state equations are Q*P = 0, and sum(P) = 1.
    # Q is (number_of_vertices x number_of_vertices).
    # We can form an augmented system.
    # The original code used number_of_vertices rows from augmented_state_matrix (which is Q.T)
    # and then the sum row.

    # The matrix for lsqr should be (number_of_equations, number_of_variables)
    # Variables are the steady_state_probabilities (number_of_vertices)
    # Equations: N flow equations (implicitly dP/dt=0) + 1 sum equation

    # Let's use the transpose of the generator matrix Q for the flow equations.
    # Q_ij is rate from j to i (i != j), Q_ii = -sum(rates out of i)
    # augmented_state_matrix as constructed is effectively Q.T (for the upper NxN block)

    equation_matrix = augmented_state_matrix  # This matrix is (N+1) x N
    # First N rows: Q.T * P (should be 0)
    # Last row: sum(P) (should be 1)

    return csc_array(equation_matrix), target_vector


def compute_average_mark_numbers(vertices_np: np.ndarray, steady_state_probabilities: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:  # Return type for average_markings changed to np.ndarray
    """
    Calculates the average number of marks (tokens) for each place in the Petri Net.

    Args:
        vertices_np (np.ndarray): 2D NumPy array of reachable markings (states).
        steady_state_probabilities (np.ndarray): Array of steady-state probabilities for each state.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mark density matrix
                                       and the NumPy array of average mark numbers for each place.
    """
    # vertices_np is expected to be a 2D NumPy array (num_states, num_places)

    # Ensure steady_state_probabilities is a column vector for broadcasting
    prob_vector_col = steady_state_probabilities.reshape(-1, 1)

    # Weighted sum of markings: P_i * M_i for each state i
    # This directly gives the average number of tokens in each place
    # E[tokens_in_place_j] = sum_i (Prob_i * tokens_in_place_j_at_state_i)
    average_tokens_per_place = np.sum(vertices_np * prob_vector_col, axis=0)

    # For marking_density_matrix (token probability density function)
    # It shows P(place_j has k tokens)
    # This requires finding unique token counts and is more complex than just average.
    # The original code's calculation for marking_density_matrix:
    # distinct_tokens = np.unique(all_markings)
    # token_presence = (all_markings[:, :, np.newaxis] == distinct_tokens)
    # marking_density_matrix = np.sum(token_presence * prob_vector[:, np.newaxis, :], axis=0)
    # This seems to calculate P(token_value_k is present in place_j), summed over states.
    # Let's refine or clarify its meaning if possible, or keep as is if it's a specific desired metric.
    # Assuming all_markings is vertices_np

    distinct_token_values = np.unique(vertices_np)  # Unique token counts across all places and states
    num_places = vertices_np.shape[1]
    num_distinct_tokens = len(distinct_token_values)

    marking_density_matrix = np.zeros((num_places, num_distinct_tokens), dtype=float)

    for place_idx in range(num_places):
        for token_val_idx, token_val in enumerate(distinct_token_values):
            # Find states where place_idx has token_val
            states_with_token_val = (vertices_np[:, place_idx] == token_val)
            # Sum probabilities of these states
            marking_density_matrix[place_idx, token_val_idx] = np.sum(steady_state_probabilities[states_with_token_val])

    # The above is P(place j has k tokens).
    # The original code's `marking_density_matrix` was (num_places, num_distinct_tokens_overall).
    # If distinct_tokens are e.g. [0,1,2,3], then marking_density_matrix[place_j, k_idx] is P(place_j has distinct_tokens[k_idx] tokens)

    return marking_density_matrix, average_tokens_per_place


def generate_stochastic_graphical_net_task(
        vertices_list: list[np.ndarray],
        edges_list: list[list[int]],
        arc_transition_indices: list[int],
        number_of_transitions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:  # Added success flag
    """
    Generates an SGN task: calculates steady-state probabilities and average mark numbers.

    Args:
        vertices_list (list[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (list[list[int]]): List of edges.
        arc_transition_indices (list[int]): Transition indices for each edge.
        number_of_transitions (int): Total number of transitions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        Steady-state probabilities, mark density, average mark numbers, lambda rates, and success flag.
        Returns None for array types if unsuccessful.
    """
    transition_rates = np.random.randint(1, 11, size=number_of_transitions).astype(float)

    # Convert vertices_list (list of 1D arrays) to a 2D NumPy array for calculations
    if not vertices_list:  # Should not happen if called after ArrGra.get_arr_gra with valid results
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
        # Using lsmr as it's generally robust for least-squares problems
        # lsqr can also be used.
        lsmr_result = lsmr(state_matrix_sparse, target_vector_eq, atol=1e-7, btol=1e-7, conlim=1e7, maxiter=None)
        # lsmr_result fields: 'x', 'istop', 'itn', 'normr', 'normar', 'norma', 'conda', 'normx'

        if lsmr_result[1] in [1, 2]:  # Solution found (exact or least-squares)
            steady_state_probabilities = lsmr_result[0]
            # Normalize probabilities to sum to 1 and be non-negative (numerical precision issues)
            steady_state_probabilities[steady_state_probabilities < 0] = 0
            if np.sum(steady_state_probabilities) > 1e-9:  # Avoid division by zero if all are zero
                steady_state_probabilities /= np.sum(steady_state_probabilities)
            else:  # All probabilities effectively zero, indicates an issue
                return None, None, None, transition_rates, False

            marking_density_array, average_markings_array = compute_average_mark_numbers(
                vertices_np, steady_state_probabilities
            )
            success = True
        else:
            # print(f"LSMR solver did not converge well. istop: {lsmr_result[1]}")
            pass  # success remains False

    except np.linalg.LinAlgError as e:
        # print(f"Linear algebra error during steady-state calculation: {e}")
        pass  # success remains False
    except ValueError as ve:  # Can happen if matrix dimensions are inconsistent
        # print(f"Value error during steady-state calculation: {ve}")
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
    if petri_net_matrix.size == 0: return False  # Handle empty matrix
    num_places = petri_net_matrix.shape[0]
    if num_places == 0: return False

    num_cols = petri_net_matrix.shape[1]
    if num_cols < 3: return False  # Needs at least one transition and marking column

    num_transitions = (num_cols - 1) // 2
    if num_transitions == 0: return False  # No transitions means not meaningfully connected in this context

    # Check if any place has no connections (sum of its row in pre/post matrices is 0)
    # Excludes the initial marking column
    if np.any(np.sum(petri_net_matrix[:, :2 * num_transitions], axis=1) == 0):
        return False

    # Check if any transition has no connections (sum of its columns in pre/post is 0)
    pre_matrix_sum_cols = np.sum(petri_net_matrix[:, :num_transitions], axis=0)
    post_matrix_sum_cols = np.sum(petri_net_matrix[:, num_transitions:2 * num_transitions], axis=0)
    if np.any(pre_matrix_sum_cols + post_matrix_sum_cols == 0):
        return False

    return True


def filter_stochastic_petri_net(
        petri_net_matrix_input: np.ndarray, place_upper_bound: int = 10, marks_lower_limit: int = 4,
        marks_upper_limit: int = 500
) -> Tuple[Dict[str, Any], bool]:
    """
    Filters a Stochastic Petri Net based on reachability and other criteria.
    Returns dictionary with NumPy arrays for relevant fields.

    Args:
        petri_net_matrix_input (np.ndarray): The Petri Net matrix.
        place_upper_bound (int): Upper bound for tokens in any single place.
        marks_lower_limit (int): Lower limit for the number of reachable markings.
        marks_upper_limit (int): Upper limit for the number of reachable markings.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing the results dictionary
                                     (with NumPy arrays) and a boolean indicating success.
    """
    # Ensure petri_net_matrix_input is a NumPy array of appropriate type
    petri_net_matrix = np.array(petri_net_matrix_input, dtype=int)

    spn_results: Dict[str, Any] = {}

    if not is_connected_graph(petri_net_matrix):  # Check connectivity early
        return spn_results, False

    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.get_arr_gra(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return spn_results, False

    # vertices is a list of 1D NumPy arrays. Convert to 2D array for processing.
    vertices_np = np.array(vertices, dtype=int)

    (steady_state_probs, marking_densities_arr,
     average_markings_arr, firing_rates_arr, sgn_success) = generate_stochastic_graphical_net_task(
        vertices, edges, arc_transitions, num_transitions  # Pass list of arrays for vertices here
    )

    if not sgn_success or steady_state_probs is None:  # Check sgn_success flag
        return spn_results, False

    # Final check on sum of average markings (spn_mu)
    current_spn_mu = np.sum(average_markings_arr)
    if not (-1000 <= current_spn_mu <= 1000):  # Wider range, original was -100 to 100
        # This check might be too strict or indicative of issues in SGN calculation for some nets
        # print(f"SPN_MU out of range: {current_spn_mu}")
        return spn_results, False

    spn_results["petri_net"] = petri_net_matrix  # Already np.ndarray, int
    spn_results["arr_vlist"] = vertices_np  # np.ndarray, int (converted from list of arrays)
    spn_results["arr_edge"] = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)  # np.ndarray, int
    spn_results["arr_tranidx"] = np.array(arc_transitions, dtype=int) if arc_transitions else np.empty((0,),
                                                                                                       dtype=int)  # np.ndarray, int

    spn_results["spn_labda"] = firing_rates_arr  # np.ndarray, float (from randint, then used in float calcs)
    spn_results["spn_steadypro"] = steady_state_probs  # np.ndarray, float
    spn_results["spn_markdens"] = marking_densities_arr  # np.ndarray, float
    spn_results["spn_allmus"] = average_markings_arr  # np.ndarray, float
    spn_results["spn_mu"] = current_spn_mu  # float (NumPy float scalar)

    return spn_results, True


def generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list: list[np.ndarray],
        edges_list: list[list[int]],
        arc_transition_indices: list[int],
        transition_rates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:  # Added success flag
    """
    Generates an SGN task with given transition rates.

    Args:
        vertices_list (list[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (list[list[int]]): List of edges.
        arc_transition_indices (list[int]): Transition indices for each edge.
        transition_rates (np.ndarray): Array of firing rates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        Steady-state probabilities, mark density, average mark numbers, and success flag.
        Returns None for array types if unsuccessful.
    """
    if not vertices_list:
        return None, None, None, False
    vertices_np = np.array(vertices_list, dtype=int)

    # Ensure transition_rates is float for calculations
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
            # print(f"LSMR (given rates) did not converge well. istop: {lsmr_result[1]}")
            pass
    except np.linalg.LinAlgError as e:
        # print(f"Linear algebra error (given rates): {e}")
        pass
    except ValueError as ve:
        # print(f"Value error (given rates): {ve}")
        pass

    return steady_state_probabilities, marking_density_array, average_markings_array, success


def get_stochastic_petri_net(
        petri_net_matrix_input: np.ndarray,
        vertices_list: list[np.ndarray],  # Changed from vertices (assumed list of lists)
        edges_list: list[list[int]],
        arc_transitions_list: list[int],
        transition_rates_input: np.ndarray
) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves SPN info given structure and transition rates. Returns dict with NumPy arrays.

    Args:
        petri_net_matrix_input (np.ndarray): The Petri Net matrix.
        vertices_list (list[np.ndarray]): List of reachable markings (1D NumPy arrays).
        edges_list (list[list[int]]): List of edges.
        arc_transitions_list (list[int]): Transition indices for each edge.
        transition_rates_input (np.ndarray): Array of firing rates.

    Returns:
        Tuple[Dict[str, Any], bool]: Results dictionary and success flag.
    """
    petri_net_matrix = np.array(petri_net_matrix_input, dtype=int)
    transition_rates = np.array(transition_rates_input, dtype=float)

    spn_results: Dict[str, Any] = {}

    if not is_connected_graph(petri_net_matrix):
        return spn_results, False

    if not vertices_list:  # vertices_list comes from ArrGra, should be non-empty if valid
        return spn_results, False
    vertices_np = np.array(vertices_list, dtype=int)

    (steady_state_probs, marking_densities_arr,
     average_markings_arr, sgn_success) = generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list, edges_list, arc_transitions_list, transition_rates
    )

    if not sgn_success or steady_state_probs is None:
        return spn_results, False

    current_spn_mu = np.sum(average_markings_arr)
    # No MU check here as per original get_stochastic_petri_net, but could be added for consistency

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
        vertices, edges, arc_transitions, transition_rates  # Pass list of arrays for vertices
    )

    if not sgn_success or steady_state_probs is None:
        return spn_results, False

    current_spn_mu = np.sum(average_markings_arr)
    if not (-1000 <= current_spn_mu <= 1000):  # Original check was -100 to 100
        # print(f"SPN_MU (DS3) out of range: {current_spn_mu}")
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
