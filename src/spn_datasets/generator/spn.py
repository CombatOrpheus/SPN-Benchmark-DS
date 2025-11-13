"""
This module provides functions for analyzing Stochastic Petri Nets (SPNs),
including computing state equations, calculating average markings, and
generating SPN tasks.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import numba
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from spn_datasets.generator import arrivable_graph as ArrGra


@numba.jit(nopython=True, cache=True)
def _compute_state_equation_numba(
    num_vertices: int,
    edges: np.ndarray,
    arc_transitions: np.ndarray,
    lambda_values: np.ndarray,
) -> np.ndarray:
    """Numba-optimized core of compute_state_equation."""
    state_matrix = np.zeros((num_vertices + 1, num_vertices), dtype=np.float64)
    for i in range(len(edges)):
        edge = edges[i]
        trans_idx = arc_transitions[i]
        src_idx, dest_idx = edge[0], edge[1]
        rate = lambda_values[trans_idx]
        state_matrix[src_idx, src_idx] -= rate
        state_matrix[dest_idx, src_idx] += rate
    state_matrix[num_vertices, :] = 1.0
    return state_matrix


def compute_state_equation(
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    lambda_values: np.ndarray,
) -> Tuple[csc_array, np.ndarray]:
    """Computes the state equation for the SPN using sparse matrices.

    Args:
        vertices: A list of reachable markings (states).
        edges: A list of edges representing transitions between states.
        arc_transitions: A list of transition indices for each edge.
        lambda_values: An array of firing rates for each transition.

    Returns:
        A tuple containing the sparse matrix for the system of equations
        and the target vector.
    """
    num_vertices = len(vertices)

    edges_arr = np.array(edges, dtype=np.int32)
    if edges_arr.ndim == 1:
        edges_arr = edges_arr.reshape(-1, 2)

    state_matrix_np = _compute_state_equation_numba(
        num_vertices,
        edges_arr,
        np.array(arc_transitions, dtype=np.int32),
        lambda_values,
    )

    state_matrix = csc_array(state_matrix_np)

    target_vector = np.zeros(num_vertices + 1, dtype=float)
    target_vector[num_vertices] = 1.0

    return state_matrix, target_vector


@numba.jit(nopython=True, cache=True)
def compute_average_markings(
    vertices: np.ndarray, steady_state_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the average number of tokens for each place.

    Args:
        vertices: A 2D array of reachable markings.
        steady_state_probs: An array of steady-state probabilities.

    Returns:
        A tuple containing the marking density matrix and the average tokens per place.
    """
    prob_col_vector = steady_state_probs.reshape(-1, 1)
    avg_tokens_per_place = np.sum(vertices * prob_col_vector, axis=0)

    unique_token_values = np.unique(vertices)
    num_places = vertices.shape[1]
    marking_density_matrix = np.zeros(
        (num_places, len(unique_token_values)), dtype=float
    )

    for place_idx in range(num_places):
        for token_idx, token_val in enumerate(unique_token_values):
            states_with_token = vertices[:, place_idx] == token_val
            marking_density_matrix[place_idx, token_idx] = np.sum(
                steady_state_probs[states_with_token]
            )

    return marking_density_matrix, avg_tokens_per_place


def solve_for_steady_state(
    state_matrix: csc_array, target_vector: np.ndarray
) -> Optional[np.ndarray]:
    """Solves for steady-state probabilities using spsolve on a modified system."""
    A_sq = state_matrix[1:, :]
    b_sq = target_vector[1:]

    try:
        probs = spsolve(A_sq, b_sq)
        probs[probs < 0] = 0
        prob_sum = np.sum(probs)
        if prob_sum > 1e-9:
            return probs / prob_sum
    except (np.linalg.LinAlgError, ValueError):
        pass

    return None


def _run_spn_task(
    vertices: list,
    edges: list,
    arc_transitions: list,
    transition_rates: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Helper to run a single SPN task."""
    if not vertices:
        return None, None, None, False

    vertices_np = np.array(vertices, dtype=int)
    state_matrix, target_vector = compute_state_equation(
        vertices, edges, arc_transitions, transition_rates
    )
    steady_state_probs = solve_for_steady_state(state_matrix, target_vector)

    if steady_state_probs is None:
        return None, None, None, False

    marking_density, avg_markings = compute_average_markings(
        vertices_np, steady_state_probs
    )
    return steady_state_probs, marking_density, avg_markings, True


def generate_stochastic_net_task(
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    num_transitions: int,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    bool,
]:
    """Generates an SPN task with random firing rates."""
    transition_rates = np.random.randint(1, 11, size=num_transitions).astype(float)
    probs, density, markings, success = _run_spn_task(
        vertices, edges, arc_transitions, transition_rates
    )
    return probs, density, markings, transition_rates, success


def generate_stochastic_net_task_with_rates(
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    transition_rates: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Generates an SPN task with specified firing rates."""
    return _run_spn_task(
        vertices, edges, arc_transitions, np.array(transition_rates, dtype=float)
    )


@numba.jit(nopython=True, cache=True)
def is_connected(petri_net_matrix: np.ndarray) -> bool:
    """Checks if the Petri net has isolated places or transitions."""
    if petri_net_matrix.size == 0 or petri_net_matrix.ndim != 2:
        return False
    num_places, num_cols = petri_net_matrix.shape
    if num_places == 0 or num_cols < 3:
        return False
    num_transitions = (num_cols - 1) // 2
    if num_transitions == 0:
        return False

    if np.any(np.sum(petri_net_matrix[:, : 2 * num_transitions], axis=1) == 0):
        return False

    pre_sum = np.sum(petri_net_matrix[:, :num_transitions], axis=0)
    post_sum = np.sum(
        petri_net_matrix[:, num_transitions : 2 * num_transitions], axis=0
    )
    if np.any(pre_sum + post_sum == 0):
        return False

    return True


def _create_spn_result_dict(
    petri_net_matrix: np.ndarray,
    vertices: list,
    edges: list,
    arc_transitions: list,
    firing_rates: np.ndarray,
    steady_state_probs: np.ndarray,
    marking_densities: np.ndarray,
    average_markings: np.ndarray,
) -> Dict[str, Any]:
    """Creates a dictionary for SPN results."""
    return {
        "petri_net": petri_net_matrix,
        "arr_vlist": np.array(vertices, dtype=int),
        "arr_edge": np.array(edges, dtype=int)
        if edges
        else np.empty((0, 2), dtype=int),
        "arr_tranidx": np.array(arc_transitions, dtype=int)
        if arc_transitions
        else np.empty((0,), dtype=int),
        "spn_labda": firing_rates,
        "spn_steadypro": steady_state_probs,
        "spn_markdens": marking_densities,
        "spn_allmus": average_markings,
        "spn_mu": np.sum(average_markings),
    }


def filter_spn(
    petri_net_matrix: np.ndarray,
    place_upper_bound: int = 10,
    marks_lower_limit: int = 4,
    marks_upper_limit: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """Filters an SPN based on reachability and other criteria."""
    petri_net_matrix = np.array(petri_net_matrix)
    if not is_connected(petri_net_matrix):
        return {}, False

    (
        vertices,
        edges,
        arc_transitions,
        num_transitions,
        is_bounded,
    ) = ArrGra.generate_reachability_graph(
        petri_net_matrix, place_upper_bound, marks_upper_limit
    )

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return {}, False

    probs, density, markings, rates, success = generate_stochastic_net_task(
        vertices, edges, arc_transitions, num_transitions
    )

    if not success or not (-1000 < np.sum(markings) < 1000):
        return {}, False

    return (
        _create_spn_result_dict(
            petri_net_matrix,
            vertices,
            edges,
            arc_transitions,
            rates,
            probs,
            density,
            markings,
        ),
        True,
    )


def get_spn_info(
    petri_net_matrix: np.ndarray,
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    transition_rates: np.ndarray,
) -> Tuple[Dict[str, Any], bool]:
    """Retrieves SPN info for a given structure and rates."""
    petri_net_matrix = np.array(petri_net_matrix)
    if not is_connected(petri_net_matrix) or not vertices:
        return {}, False

    probs, density, markings, success = generate_stochastic_net_task_with_rates(
        vertices, edges, arc_transitions, transition_rates
    )

    if not success:
        return {}, False

    return (
        _create_spn_result_dict(
            petri_net_matrix,
            vertices,
            edges,
            arc_transitions,
            transition_rates,
            probs,
            density,
            markings,
        ),
        True,
    )
