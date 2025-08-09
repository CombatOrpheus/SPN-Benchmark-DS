#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : SPN.py
@Date    : 2020-08-22 (modified for optimized NumPy and sparse matrix construction)
@Author  : mingjian

This module provides functionalities to work with Stochastic Petri Nets (SPNs).
It includes methods for:
- Computing the state equation of an SPN.
- Calculating steady-state probabilities and average token markings.
- Generating and filtering SPN models based on structural and behavioral properties.
- Solving for SPN properties given specific transition rates.
"""

from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy.sparse import csc_array, lil_matrix
from scipy.sparse.linalg import lsmr

from DataGenerate import ArrivableGraph as ArrGra


def compute_state_equation(
        vertices: List[np.ndarray],
        edges: List[List[int]],
        arc_transitions: List[int],
        lambda_values: np.ndarray
) -> Tuple[csc_array, np.ndarray]:
    """
    Calculates the state equation for the Stochastic Petri Net using sparse matrices.

    Args:
        vertices: List of reachable markings (states), each marking is a 1D NumPy array.
        edges: List of edges representing transitions between states.
        arc_transitions: List of transition indices corresponding to each edge.
        lambda_values: Array of firing rates for each transition.

    Returns:
        A tuple containing the sparse matrix (CSC format) for the system of equations
        and the target vector for the solver.
    """
    number_of_vertices = len(vertices)

    # Initialize a LIL (List of Lists) sparse matrix for efficient construction.
    # The matrix is augmented with an extra row for the probability summation constraint.
    augmented_state_matrix = lil_matrix((number_of_vertices + 1, number_of_vertices), dtype=float)

    # Populate the matrix based on transition rates between states.
    for edge, transition_index in zip(edges, arc_transitions):
        source_vertex_idx, dest_vertex_idx = edge
        rate = lambda_values[transition_index]
        augmented_state_matrix[source_vertex_idx, source_vertex_idx] -= rate
        augmented_state_matrix[dest_vertex_idx, source_vertex_idx] += rate

    # Add the constraint that the sum of all steady-state probabilities must equal 1.
    augmented_state_matrix[number_of_vertices, :] = 1.0

    # The target vector is zero everywhere except for the constraint equation.
    target_vector = np.zeros(number_of_vertices + 1, dtype=float)
    target_vector[number_of_vertices] = 1.0

    # Convert to CSC (Compressed Sparse Column) format for efficient matrix-vector products by solvers.
    return augmented_state_matrix.tocsc(), target_vector


def compute_average_mark_numbers(
        vertices_np: np.ndarray,
        steady_state_probabilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the average number of tokens for each place in the Petri Net in a vectorized manner.

    Args:
        vertices_np: A 2D NumPy array of reachable markings (states), where each row is a state.
        steady_state_probabilities: A 1D array of steady-state probabilities for each state.

    Returns:
        A tuple containing:
        - The marking density matrix.
        - A NumPy array of the average number of tokens for each place.
    """
    # Average tokens = sum(marking * probability) for each place.
    average_tokens_per_place = np.sum(vertices_np * steady_state_probabilities.reshape(-1, 1), axis=0)

    # Calculate the probability density of markings for each place using vectorized operations.
    num_places = vertices_np.shape[1]

    distinct_token_values = np.unique(vertices_np)
    token_map = {token: i for i, token in enumerate(distinct_token_values)}

    # Vectorize the mapping from token values to indices
    indexed_vertices = np.vectorize(token_map.get)(vertices_np)

    marking_density_matrix = np.zeros((num_places, len(distinct_token_values)), dtype=float)

    for place_idx in range(num_places):
        # Use bincount to sum probabilities for each token value in the current place.
        # It's much faster than iterating and summing in Python.
        marking_density_matrix[place_idx] = np.bincount(
            indexed_vertices[:, place_idx],
            weights=steady_state_probabilities,
            minlength=len(distinct_token_values)
        )

    return marking_density_matrix, average_tokens_per_place


def _solve_state_equation(
        state_matrix: csc_array,
        target_vector: np.ndarray
) -> Optional[np.ndarray]:
    """
    Internal function to solve the system of linear equations for steady-state probabilities.

    Args:
        state_matrix: The sparse CSC matrix of the state equations.
        target_vector: The target vector for the solver.

    Returns:
        A NumPy array of steady-state probabilities if a solution is found, otherwise None.
    """
    try:
        # lsmr is an iterative solver for sparse linear systems.
        lsmr_result = lsmr(state_matrix, target_vector, atol=1e-7, btol=1e-7, conlim=1e7, maxiter=None)

        # istop (lsmr_result[1]) indicates the outcome.
        # 1: x is an approximate solution to A*x = b.
        # 2: x approximately solves the least-squares problem.
        if lsmr_result[1] in [1, 2]:
            steady_state_probs = lsmr_result[0]
            # Ensure probabilities are non-negative.
            steady_state_probs[steady_state_probs < 0] = 0
            # Normalize the probability vector to sum to 1.
            prob_sum = np.sum(steady_state_probs)
            if prob_sum > 1e-9:
                return steady_state_probs / prob_sum
            else:
                return None  # Sum of probabilities is too small.
    except (np.linalg.LinAlgError, ValueError):
        # Errors during solving are caught and treated as failure.
        pass
    return None


def generate_stochastic_graphical_net_task(
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transition_indices: List[int],
        number_of_transitions: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, bool]:
    """
    Generates an SGN task: calculates steady-state probabilities and average mark numbers.
    Transition rates are generated randomly.

    Args:
        vertices_list: List of reachable markings (1D NumPy arrays).
        edges_list: List of edges between markings.
        arc_transition_indices: Transition indices corresponding to each edge.
        number_of_transitions: Total number of transitions in the Petri net.

    Returns:
        A tuple containing:
        - Steady-state probabilities (or None on failure).
        - Mark density matrix (or None on failure).
        - Average mark numbers (or None on failure).
        - The randomly generated lambda (firing) rates.
        - A boolean success flag.
    """
    transition_rates = np.random.randint(1, 11, size=number_of_transitions).astype(float)

    if not vertices_list:
        return None, None, None, transition_rates, False

    state_matrix, target_vector = compute_state_equation(
        vertices_list, edges_list, arc_transition_indices, transition_rates
    )

    steady_state_probs = _solve_state_equation(state_matrix, target_vector)

    if steady_state_probs is None:
        return None, None, None, transition_rates, False

    vertices_np = np.array(vertices_list, dtype=int)
    marking_density, average_markings = compute_average_mark_numbers(vertices_np, steady_state_probs)
    return steady_state_probs, marking_density, average_markings, transition_rates, True


def generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transition_indices: List[int],
        transition_rates: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Generates an SGN task with pre-defined transition rates.

    Args:
        vertices_list: List of reachable markings.
        edges_list: List of edges between markings.
        arc_transition_indices: Transition indices for each edge.
        transition_rates: The firing rates for each transition.

    Returns:
        A tuple containing:
        - Steady-state probabilities (or None on failure).
        - Mark density matrix (or None on failure).
        - Average mark numbers (or None on failure).
        - A boolean success flag.
    """
    if not vertices_list:
        return None, None, None, False

    state_matrix, target_vector = compute_state_equation(
        vertices_list, edges_list, arc_transition_indices, transition_rates
    )

    steady_state_probs = _solve_state_equation(state_matrix, target_vector)

    if steady_state_probs is None:
        return None, None, None, False

    vertices_np = np.array(vertices_list, dtype=int)
    marking_density, average_markings = compute_average_mark_numbers(vertices_np, steady_state_probs)
    return steady_state_probs, marking_density, average_markings, True


def is_connected_graph(petri_net_matrix: np.ndarray) -> bool:
    """
    Checks if the Petri Net graph has isolated places or transitions.
    An isolated node has no input or output arcs.

    Args:
        petri_net_matrix: The matrix representation of the Petri Net, structured as
                          [Pre-matrix | Post-matrix | Initial Marking].

    Returns:
        True if the graph is connected (no isolated nodes), False otherwise.
    """
    if petri_net_matrix.size == 0:
        return False
    num_places, num_cols = petri_net_matrix.shape
    if num_places == 0 or num_cols < 3:
        return False

    num_transitions = (num_cols - 1) // 2
    if num_transitions == 0:
        return False

    # Check for isolated places (rows in the pre/post matrix that sum to zero).
    io_matrix = petri_net_matrix[:, :2 * num_transitions]
    if np.any(np.sum(io_matrix, axis=1) == 0):
        return False

    # Check for isolated transitions (columns in the pre/post matrix that sum to zero).
    pre_matrix = petri_net_matrix[:, :num_transitions]
    post_matrix = petri_net_matrix[:, num_transitions:2 * num_transitions]
    if np.any(np.sum(pre_matrix, axis=0) + np.sum(post_matrix, axis=0) == 0):
        return False

    return True


def _build_spn_result_dict(
        petri_net_matrix: np.ndarray,
        vertices: List[np.ndarray],
        edges: List[List[int]],
        arc_transitions: List[int],
        firing_rates: np.ndarray,
        steady_state_probs: np.ndarray,
        marking_densities: np.ndarray,
        average_markings: np.ndarray
) -> Dict[str, Any]:
    """Constructs the dictionary of SPN results."""
    return {
        "petri_net": petri_net_matrix,
        "arr_vlist": np.array(vertices, dtype=int),
        "arr_edge": np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int),
        "arr_tranidx": np.array(arc_transitions, dtype=int) if arc_transitions else np.empty((0,), dtype=int),
        "spn_labda": firing_rates,
        "spn_steadypro": steady_state_probs,
        "spn_markdens": marking_densities,
        "spn_allmus": average_markings,
        "spn_mu": np.sum(average_markings)
    }


def filter_stochastic_petri_net(
        petri_net_matrix: np.ndarray,
        place_upper_bound: int = 10,
        marks_lower_limit: int = 4,
        marks_upper_limit: int = 500
) -> Tuple[Dict[str, Any], bool]:
    """
    Filters a Stochastic Petri Net based on reachability and solvability criteria.

    Args:
        petri_net_matrix: The matrix representation of the Petri Net.
        place_upper_bound: Maximum tokens allowed in any single place during reachability analysis.
        marks_lower_limit: Minimum number of reachable markings for the SPN to be valid.
        marks_upper_limit: Maximum number of reachable markings to explore.

    Returns:
        A tuple containing the results dictionary and a boolean indicating success.
    """
    if not is_connected_graph(petri_net_matrix):
        return {}, False

    # Generate the reachability graph.
    vertices, edges, arc_transitions, num_transitions, is_bounded = ArrGra.build_reachability_graph(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return {}, False

    # Solve for SPN properties with random firing rates.
    task_results = generate_stochastic_graphical_net_task(
        vertices, edges, arc_transitions, num_transitions
    )
    steady_probs, mark_densities, avg_markings, firing_rates, success = task_results

    if not success:
        return {}, False

    # Final stability check on the average number of tokens.
    total_avg_mu = np.sum(avg_markings)
    if not (-1000 <= total_avg_mu <= 1000):
        return {}, False

    spn_results = _build_spn_result_dict(
        petri_net_matrix, vertices, edges, arc_transitions,
        firing_rates, steady_probs, mark_densities, avg_markings
    )
    return spn_results, True


def get_stochastic_petri_net(
        petri_net_matrix_input: np.ndarray,
        vertices_list: List[np.ndarray],
        edges_list: List[List[int]],
        arc_transitions_list: List[int],
        transition_rates_input: np.ndarray
) -> Tuple[Dict[str, Any], bool]:
    """
    Retrieves SPN properties for a given structure and pre-defined transition rates.

    Args:
        petri_net_matrix_input: The Petri Net matrix.
        vertices_list: List of reachable markings.
        edges_list: List of edges between markings.
        arc_transitions_list: Transition indices for each edge.
        transition_rates_input: The firing rates for each transition.

    Returns:
        A tuple containing the results dictionary and a boolean indicating success.
    """
    if not is_connected_graph(petri_net_matrix_input) or not vertices_list:
        return {}, False

    # Solve for SPN properties with the given firing rates.
    task_results = generate_stochastic_graphical_net_task_with_given_rates(
        vertices_list, edges_list, arc_transitions_list, transition_rates_input
    )
    steady_probs, mark_densities, avg_markings, success = task_results

    if not success:
        return {}, False

    spn_results = _build_spn_result_dict(
        petri_net_matrix_input, vertices_list, edges_list, arc_transitions_list,
        transition_rates_input, steady_probs, mark_densities, avg_markings
    )
    return spn_results, True


def get_spnds3(
        petri_net_matrix_input: np.ndarray,
        transition_rates_input: np.ndarray,
        place_upper_bound: int = 10,
        marks_lower_limit: int = 4,
        marks_upper_limit: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """
    Generates and solves an SPN for a specific configuration (e.g., for "DS3" dataset).
    This function first builds the reachability graph and then solves the SPN.

    Args:
        petri_net_matrix_input: The Petri Net matrix.
        transition_rates_input: The firing rates for each transition.
        place_upper_bound: Max tokens per place for reachability analysis.
        marks_lower_limit: Min number of reachable markings.
        marks_upper_limit: Max number of reachable markings to explore.

    Returns:
        A tuple containing the results dictionary and a boolean indicating success.
    """
    if not is_connected_graph(petri_net_matrix_input):
        return {}, False

    # Generate the reachability graph.
    vertices, edges, arc_transitions, _, is_bounded = ArrGra.build_reachability_graph(
        petri_net_matrix_input, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return {}, False

    # Solve for SPN properties with the given firing rates.
    task_results = generate_stochastic_graphical_net_task_with_given_rates(
        vertices, edges, arc_transitions, transition_rates_input
    )
    steady_probs, mark_densities, avg_markings, success = task_results

    if not success:
        return {}, False

    # Final stability check.
    total_avg_mu = np.sum(avg_markings)
    if not (-1000 <= total_avg_mu <= 1000):
        return {}, False

    spn_results = _build_spn_result_dict(
        petri_net_matrix_input, vertices, edges, arc_transitions,
        transition_rates_input, steady_probs, mark_densities, avg_markings
    )
    return spn_results, True
