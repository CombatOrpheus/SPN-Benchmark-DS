"""
This module provides functions for analyzing Stochastic Petri Nets (SPNs),
including computing state equations, calculating average markings, and
generating SPN tasks.
"""

from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import warnings
import numpy as np
import numba
from scipy.sparse import csc_array, lil_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from spn_datasets.generator import ArrivableGraph as ArrGra


@numba.jit(nopython=True, cache=True)
def _compute_state_equation_numba(num_vertices, edges, arc_transitions, lambda_values):
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

    # Use Numba-optimized function for the core computation
    edges_arr = np.array(edges, dtype=np.int32)
    if edges_arr.ndim == 1:
        # Ensure that the array is 2D, even if empty
        edges_arr = edges_arr.reshape(-1, 2)

    state_matrix_np = _compute_state_equation_numba(
        num_vertices,
        edges_arr,
        np.array(arc_transitions, dtype=np.int32),
        lambda_values,
    )

    # Convert the NumPy array to a sparse matrix
    state_matrix = csc_array(state_matrix_np)

    target_vector = np.zeros(num_vertices + 1, dtype=float)
    target_vector[num_vertices] = 1.0

    return state_matrix, target_vector


@numba.jit(nopython=True, cache=True)
def compute_average_markings(vertices: np.ndarray, steady_state_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    marking_density_matrix = np.zeros((num_places, len(unique_token_values)), dtype=float)

    for place_idx in range(num_places):
        for token_idx, token_val in enumerate(unique_token_values):
            states_with_token = vertices[:, place_idx] == token_val
            marking_density_matrix[place_idx, token_idx] = np.sum(steady_state_probs[states_with_token])

    return marking_density_matrix, avg_tokens_per_place


def solve_for_steady_state(state_matrix: csc_array, target_vector: np.ndarray) -> np.ndarray:
    """Solves for steady-state probabilities using spsolve on a modified system."""
    num_vertices = state_matrix.shape[1]

    # To use spsolve, we need a square matrix. We can achieve this by removing
    # one of the redundant equations from the state matrix (the first `num_vertices` rows).
    # We remove the first row to make it a square matrix of size (num_vertices, num_vertices).
    A_sq = state_matrix[1:, :]
    b_sq = target_vector[1:]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            probs = spsolve(A_sq, b_sq)

        # Normalize probabilities
        probs[probs < 0] = 0
        prob_sum = np.sum(probs)
        if prob_sum > 1e-9:
            return probs / prob_sum

    except (np.linalg.LinAlgError, ValueError, MatrixRankWarning):
        pass  # Handle numerical issues

    return None


def _run_sgn_task(
    vertices, edges, arc_transitions, transition_rates
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Helper to run a single SGN task."""
    if not vertices:
        return None, None, None, False

    vertices_np = np.array(vertices, dtype=int)
    state_matrix, target_vector = compute_state_equation(vertices, edges, arc_transitions, transition_rates)
    steady_state_probs = solve_for_steady_state(state_matrix, target_vector)

    if steady_state_probs is None:
        return None, None, None, False

    marking_density, avg_markings = compute_average_markings(vertices_np, steady_state_probs)
    return steady_state_probs, marking_density, avg_markings, True


def generate_stochastic_net_task(
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    num_transitions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generates an SGN task with random firing rates.

    Returns:
        A tuple with steady-state probabilities, mark density, average markings,
        firing rates, and a success flag.
    """
    transition_rates = np.random.randint(1, 11, size=num_transitions).astype(float)
    probs, density, markings, success = _run_sgn_task(vertices, edges, arc_transitions, transition_rates)
    return probs, density, markings, transition_rates, success


def generate_stochastic_net_task_with_rates(
    vertices: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    transition_rates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generates an SGN task with specified firing rates."""
    return _run_sgn_task(vertices, edges, arc_transitions, np.array(transition_rates, dtype=float))


@numba.jit(nopython=True, cache=True)
def is_connected(petri_net_matrix):
    """Checks if the Petri net is weakly connected (single component).

    This function treats the Petri net as a bipartite graph (Places U Transitions)
    and checks if all nodes belong to a single connected component using BFS.
    It considers edges as undirected for connectivity check.

    Args:
        petri_net_matrix (np.ndarray): The Petri net matrix of shape
            (num_places, 2 * num_transitions + 1).

    Returns:
        bool: True if the graph is connected, False otherwise.
    """
    if petri_net_matrix.size == 0:
        return False
    if petri_net_matrix.ndim != 2:
        return False
    num_places, num_cols = petri_net_matrix.shape
    if num_places == 0 or num_cols < 3:
        return False
    num_transitions = (num_cols - 1) // 2
    if num_transitions == 0:
        return False

    # Check for isolated places (fast fail)
    if np.any(np.sum(petri_net_matrix[:, : 2 * num_transitions], axis=1) == 0):
        return False

    # Check for isolated transitions (fast fail)
    pre_sum = np.sum(petri_net_matrix[:, :num_transitions], axis=0)
    post_sum = np.sum(petri_net_matrix[:, num_transitions : 2 * num_transitions], axis=0)
    if np.any(pre_sum + post_sum == 0):
        return False

    # BFS to check for full connectivity (single component)
    num_nodes = num_places + num_transitions
    visited = np.zeros(num_nodes, dtype=np.bool_)
    queue = np.empty(num_nodes, dtype=np.int32)
    head = 0
    tail = 0

    # Start BFS from node 0 (which is a Place)
    queue[tail] = 0
    tail += 1
    visited[0] = True
    count = 0

    while head < tail:
        u = queue[head]
        head += 1
        count += 1

        if u < num_places:
            # u is a Place
            p = u
            for t in range(num_transitions):
                # Check connection P -> T (Pre) or T -> P (Post)
                # Pre is at column t, Post is at column num_transitions + t
                if petri_net_matrix[p, t] == 1 or petri_net_matrix[p, num_transitions + t] == 1:
                    v = num_places + t
                    if not visited[v]:
                        visited[v] = True
                        queue[tail] = v
                        tail += 1
        else:
            # u is a Transition
            t = u - num_places
            for p in range(num_places):
                if petri_net_matrix[p, t] == 1 or petri_net_matrix[p, num_transitions + t] == 1:
                    v = p
                    if not visited[v]:
                        visited[v] = True
                        queue[tail] = v
                        tail += 1

    return count == num_nodes


def compute_qualitative_properties(
    vertices: List[np.ndarray],
    edges: List[List[int]],
) -> Dict[str, Any]:
    """Computes qualitative properties of the SPN reachability graph.

    Metrics:
    - is_deadlock_free: True if every reachable state has at least one enabled transition.
    - is_reversible: True if the initial marking is reachable from every reachable state.
    - is_safe: True if the number of tokens in any place never exceeds 1.
    - max_tokens: The maximum number of tokens in any place across all reachable markings.
    """
    if not vertices:
        return {
            "is_deadlock_free": False,
            "is_reversible": False,
            "is_safe": False,
            "max_tokens": 0,
        }

    num_vertices = len(vertices)

    # Max tokens and Safeness
    vertices_arr = np.array(vertices)
    max_tokens = int(np.max(vertices_arr))
    is_safe = max_tokens <= 1

    if not edges:
        # If there are vertices but no edges, and num_vertices > 0
        # If only 1 state (initial) and no edges, it's a deadlock unless no transitions exist.
        return {
            "is_deadlock_free": False,
            "is_reversible": True if num_vertices == 1 else False,
            "is_safe": is_safe,
            "max_tokens": max_tokens,
        }

    # Deadlock-free: Check if every node has an outgoing edge
    sources = set(e[0] for e in edges)
    is_deadlock_free = len(sources) == num_vertices

    # Reversibility: Check if M0 (index 0) is reachable from all nodes
    # We do a BFS on the transpose graph starting from 0.
    adj_t = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adj_t[v].append(u)

    visited = [False] * num_vertices
    queue = deque([0])
    visited[0] = True
    count = 1  # visited 0

    while queue:
        u = queue.popleft()
        for v in adj_t[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                count += 1

    is_reversible = count == num_vertices

    return {
        "is_deadlock_free": is_deadlock_free,
        "is_reversible": is_reversible,
        "is_safe": is_safe,
        "max_tokens": max_tokens,
    }


def _create_spn_result_dict(
    petri_net_matrix,
    vertices,
    edges,
    arc_transitions,
    firing_rates,
    steady_state_probs,
    marking_densities,
    average_markings,
) -> Dict[str, Any]:
    """Creates a dictionary for SPN results."""
    result = {
        "petri_net": petri_net_matrix,
        "arr_vlist": np.array(vertices, dtype=int),
        "arr_edge": np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int),
        "arr_tranidx": np.array(arc_transitions, dtype=int) if arc_transitions else np.empty((0,), dtype=int),
        "spn_labda": firing_rates,
        "spn_steadypro": steady_state_probs,
        "spn_markdens": marking_densities,
        "spn_allmus": average_markings,
        "spn_mu": np.sum(average_markings),
    }

    qual_props = compute_qualitative_properties(vertices, edges)
    result.update(qual_props)
    return result


def filter_spn(
    petri_net_matrix: np.ndarray,
    place_upper_bound: int = 10,
    marks_lower_limit: int = 4,
    marks_upper_limit: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """Filters an SPN based on reachability and other criteria.

    Returns:
        A tuple containing the results dictionary and a success flag.
    """
    petri_net_matrix = np.array(petri_net_matrix)  # Ensure it's a numpy array
    if not is_connected(petri_net_matrix):
        return {}, False

    (
        vertices,
        edges,
        arc_transitions,
        num_transitions,
        is_bounded,
    ) = ArrGra.generate_reachability_graph(petri_net_matrix, place_upper_bound, marks_upper_limit)

    if not is_bounded or not vertices or len(vertices) < marks_lower_limit:
        return {}, False

    (
        probs,
        density,
        markings,
        rates,
        success,
    ) = generate_stochastic_net_task(vertices, edges, arc_transitions, num_transitions)

    if not success or np.sum(markings) > 1000 or np.sum(markings) < -1000:
        return {}, False

    return (
        _create_spn_result_dict(petri_net_matrix, vertices, edges, arc_transitions, rates, probs, density, markings),
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
    petri_net_matrix = np.array(petri_net_matrix)  # Ensure it's a numpy array
    if not is_connected(petri_net_matrix) or not vertices:
        return {}, False

    (
        probs,
        density,
        markings,
        success,
    ) = generate_stochastic_net_task_with_rates(vertices, edges, arc_transitions, transition_rates)

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
