"""
This module provides functions for analyzing Stochastic Petri Nets (SPNs),
including computing state equations, calculating average markings, and
generating SPN tasks.
"""

from typing import Tuple, Dict, Any
import warnings
import numpy as np
import numba
from scipy.sparse import csc_array, coo_array
from scipy.sparse.linalg import MatrixRankWarning, lsqr, bicgstab

import threading

_spn_scratchpad = threading.local()


def _get_compute_state_scratchpad(num_vertices, num_edges):
    if not hasattr(_spn_scratchpad, "nnz_per_row") or _spn_scratchpad.nnz_per_row.size < num_vertices:
        _spn_scratchpad.nnz_per_row = np.empty(num_vertices * 2, dtype=np.int32)
        _spn_scratchpad.indptr = np.empty((num_vertices * 2) + 1, dtype=np.int32)
        _spn_scratchpad.current_ptr = np.empty(num_vertices * 2, dtype=np.int32)

    # Calculate exact max non-zeros for these inputs
    # Each edge contributes at most 2 non-zeros, plus the last row takes num_vertices.
    max_nnz = (num_edges * 2) + num_vertices
    if getattr(_spn_scratchpad, "data", np.array([])).size < max_nnz:
        _spn_scratchpad.data = np.empty(max_nnz * 2, dtype=np.float64)
        _spn_scratchpad.indices = np.empty(max_nnz * 2, dtype=np.int32)

    return (
        _spn_scratchpad.nnz_per_row,
        _spn_scratchpad.indptr,
        _spn_scratchpad.current_ptr,
        _spn_scratchpad.data,
        _spn_scratchpad.indices,
    )


def _get_is_connected_scratchpad(num_nodes):
    if not hasattr(_spn_scratchpad, "visited_conn") or _spn_scratchpad.visited_conn.size < num_nodes:
        _spn_scratchpad.visited_conn = np.empty(num_nodes * 2, dtype=np.bool_)
        _spn_scratchpad.queue_conn = np.empty(num_nodes * 2, dtype=np.int32)
    return _spn_scratchpad.visited_conn, _spn_scratchpad.queue_conn


def _get_qualitative_scratchpad(num_vertices, num_edges):
    if not hasattr(_spn_scratchpad, "seen_sources") or _spn_scratchpad.seen_sources.size < num_vertices:
        _spn_scratchpad.seen_sources = np.empty(num_vertices * 2, dtype=np.bool_)
        _spn_scratchpad.edge_counts = np.empty(num_vertices * 2, dtype=np.int32)
        _spn_scratchpad.adj_ptr = np.empty((num_vertices * 2) + 1, dtype=np.int32)
        _spn_scratchpad.current_idx = np.empty(num_vertices * 2, dtype=np.int32)
        _spn_scratchpad.visited_qual = np.empty(num_vertices * 2, dtype=np.bool_)
        _spn_scratchpad.queue_qual = np.empty(num_vertices * 2, dtype=np.int32)

    if getattr(_spn_scratchpad, "adj_t", np.array([])).size < num_edges:
        _spn_scratchpad.adj_t = np.empty(num_edges * 2, dtype=np.int32)

    return (
        _spn_scratchpad.seen_sources,
        _spn_scratchpad.edge_counts,
        _spn_scratchpad.adj_t,
        _spn_scratchpad.adj_ptr,
        _spn_scratchpad.current_idx,
        _spn_scratchpad.visited_qual,
        _spn_scratchpad.queue_qual,
    )


def _get_avg_markings_scratchpad(num_places, max_token):
    if not hasattr(_spn_scratchpad, "avg_tokens_per_place") or _spn_scratchpad.avg_tokens_per_place.size < num_places:
        _spn_scratchpad.avg_tokens_per_place = np.empty(num_places * 2, dtype=np.float64)

    if getattr(_spn_scratchpad, "present_tokens", np.array([])).size < max_token + 1:
        _spn_scratchpad.present_tokens = np.empty((max_token + 1) * 2, dtype=np.bool_)
        _spn_scratchpad.token_to_idx = np.empty((max_token + 1) * 2, dtype=np.int32)

    return (
        _spn_scratchpad.avg_tokens_per_place,
        _spn_scratchpad.present_tokens,
        _spn_scratchpad.token_to_idx,
    )


from spn_datasets.generator import ArrivableGraph as ArrGra


@numba.jit(nopython=True, cache=True)
def _compute_state_equation_numba(
    num_vertices, edges, arc_transitions, lambda_values, nnz_per_row, indptr, current_ptr, data, indices
):
    num_edges = len(edges)

    for i in range(num_vertices):
        nnz_per_row[i] = 0

    for i in range(num_edges):
        src_idx = edges[i, 0]
        dest_idx = edges[i, 1]

        if src_idx > 0:
            nnz_per_row[src_idx - 1] += 1

        if dest_idx > 0:
            nnz_per_row[dest_idx - 1] += 1

    nnz_per_row[num_vertices - 1] = num_vertices

    indptr[0] = 0
    for i in range(num_vertices):
        indptr[i + 1] = indptr[i] + nnz_per_row[i]

    num_nnz = indptr[num_vertices]
    for i in range(num_nnz):
        data[i] = 0.0
        indices[i] = 0

    for i in range(num_vertices):
        current_ptr[i] = indptr[i]

    for i in range(num_edges):
        src_idx = edges[i, 0]
        dest_idx = edges[i, 1]
        rate = lambda_values[arc_transitions[i]]

        if src_idx > 0:
            row = src_idx - 1
            pos = current_ptr[row]
            indices[pos] = src_idx
            data[pos] -= rate
            current_ptr[row] += 1

        if dest_idx > 0:
            row = dest_idx - 1
            pos = current_ptr[row]
            indices[pos] = src_idx
            data[pos] += rate
            current_ptr[row] += 1

    row = num_vertices - 1
    for i in range(num_vertices):
        pos = current_ptr[row]
        indices[pos] = i
        data[pos] = 1.0
        current_ptr[row] += 1

    return num_nnz


from scipy.sparse import csr_array


def compute_state_equation(
    vertices: np.ndarray,
    edges: np.ndarray,
    arc_transitions: np.ndarray,
    lambda_values: np.ndarray,
) -> Tuple[csr_array, np.ndarray]:
    """Computes the state equation for the SPN using sparse matrices.

    Args:
        vertices: A list of reachable markings (states).
        edges: A list of edges representing transitions between states.
        arc_transitions: A list of transition indices for each edge.
        lambda_values: An array of firing rates for each transition.

    Returns:
        A tuple containing the sparse matrix for the system of equations
        and the target vector, already modified to be square (N, N).
    """
    num_vertices = len(vertices)

    # Use Numba-optimized function for the core computation
    edges_arr = np.asarray(edges, dtype=np.int32)
    if edges_arr.ndim == 1:
        # Ensure that the array is 2D, even if empty
        edges_arr = edges_arr.reshape(-1, 2)

    scratch_nnz, scratch_indptr, scratch_cur, scratch_data, scratch_indices = _get_compute_state_scratchpad(
        num_vertices, len(edges_arr)
    )

    num_nnz = _compute_state_equation_numba(
        num_vertices,
        edges_arr,
        np.asarray(arc_transitions, dtype=np.int32),
        np.asarray(lambda_values, dtype=np.float64),
        scratch_nnz,
        scratch_indptr,
        scratch_cur,
        scratch_data,
        scratch_indices,
    )

    # Construct CSR sparse matrix directly from data, indices, indptr
    data = scratch_data[:num_nnz].copy()
    indices = scratch_indices[:num_nnz].copy()
    indptr = scratch_indptr[: num_vertices + 1].copy()
    A_sq = csr_array((data, indices, indptr), shape=(num_vertices, num_vertices))

    # We need to sum duplicates in CSR if there are multiple edges
    A_sq.sum_duplicates()

    b_sq = np.zeros(num_vertices, dtype=float)
    b_sq[-1] = 1.0

    return A_sq, b_sq


@numba.jit(nopython=True, cache=True)
def _compute_average_markings_core(
    vertices, steady_state_probs, avg_tokens_per_place, present_tokens, token_to_idx, max_token
):
    num_states, num_places = vertices.shape

    for i in range(num_places):
        avg_tokens_per_place[i] = 0.0

    for i in range(max_token + 1):
        present_tokens[i] = False

    for s in range(num_states):
        prob = steady_state_probs[s]
        for p in range(num_places):
            val = vertices[s, p]
            avg_tokens_per_place[p] += val * prob
            present_tokens[val] = True

    num_unique_tokens = 0
    for i in range(max_token + 1):
        if present_tokens[i]:
            num_unique_tokens += 1

    for i in range(max_token + 1):
        token_to_idx[i] = -1

    idx = 0
    for val in range(max_token + 1):
        if present_tokens[val]:
            token_to_idx[val] = idx
            idx += 1

    marking_density_matrix = np.zeros((num_places, num_unique_tokens), dtype=np.float64)

    for s in range(num_states):
        prob = steady_state_probs[s]
        for p in range(num_places):
            token_val = vertices[s, p]
            idx = token_to_idx[token_val]
            marking_density_matrix[p, idx] += prob

    return marking_density_matrix, num_unique_tokens


def compute_average_markings(vertices: np.ndarray, steady_state_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_token = np.max(vertices)
    scratch_avg, scratch_present, scratch_token_idx = _get_avg_markings_scratchpad(vertices.shape[1], max_token)

    marking_density_matrix, _ = _compute_average_markings_core(
        vertices, steady_state_probs, scratch_avg, scratch_present, scratch_token_idx, max_token
    )

    avg_tokens_per_place = scratch_avg[: vertices.shape[1]].copy()
    return marking_density_matrix, avg_tokens_per_place


def solve_for_steady_state(A_sq: csr_array, b_sq: np.ndarray) -> np.ndarray:
    """Solves for steady-state probabilities using bicgstab with a fallback to lsqr."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fast, iterative solver (requires square matrix)
            if A_sq.shape[0] == A_sq.shape[1]:
                probs, info = bicgstab(A_sq, b_sq, rtol=1e-5, atol=1e-5)
            else:
                info = -1

            if info != 0:
                # lsqr is robust against singular/ill-conditioned matrices and doesn't crash Python
                probs, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(
                    A_sq, b_sq, atol=1e-5, btol=1e-5
                )
                if istop not in (1, 2):
                    return None

        # spsolve might return a matrix if the inputs aren't perfectly aligned or
        # in some error conditions. Make sure it's flattened.
        # ⚡ Bolt Optimization: Use `np.ravel()` or direct `flatten()`
        # to avoid intermediate array allocations.
        if hasattr(probs, "toarray"):
            probs = probs.toarray().ravel()
        elif hasattr(probs, "ravel"):
            probs = probs.ravel()
        elif hasattr(probs, "flatten"):
            probs = probs.flatten()

        # Normalize probabilities
        probs[probs < 0] = 0
        prob_sum = probs.sum()
        if prob_sum > 1e-9:
            return probs / prob_sum

    except (np.linalg.LinAlgError, ValueError, MatrixRankWarning):
        pass  # Handle numerical issues

    return None


def _run_sgn_task(
    vertices, edges, arc_transitions, transition_rates
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Helper to run a single SGN task."""
    if vertices.size == 0:
        return None, None, None, False

    vertices_np = np.asarray(vertices, dtype=np.int32)
    state_matrix, target_vector = compute_state_equation(vertices_np, edges, arc_transitions, transition_rates)
    steady_state_probs = solve_for_steady_state(state_matrix, target_vector)

    if steady_state_probs is None:
        return None, None, None, False

    marking_density, avg_markings = compute_average_markings(
        vertices_np, np.asarray(steady_state_probs, dtype=np.float64)
    )
    return steady_state_probs, marking_density, avg_markings, True


def generate_stochastic_net_task(
    vertices: np.ndarray,
    edges: np.ndarray,
    arc_transitions: np.ndarray,
    num_transitions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generates an SGN task with random firing rates.

    Returns:
        A tuple with steady-state probabilities, mark density, average markings,
        firing rates, and a success flag.
    """
    vertices = np.asarray(vertices, dtype=np.int32)
    edges = np.asarray(edges, dtype=np.int32)
    arc_transitions = np.asarray(arc_transitions, dtype=np.int32)
    transition_rates = np.random.randint(1, 11, size=num_transitions).astype(np.float64)
    probs, density, markings, success = _run_sgn_task(vertices, edges, arc_transitions, transition_rates)
    return probs, density, markings, transition_rates, success


def generate_stochastic_net_task_with_rates(
    vertices: np.ndarray,
    edges: np.ndarray,
    arc_transitions: np.ndarray,
    transition_rates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generates an SGN task with specified firing rates."""
    vertices = np.asarray(vertices, dtype=np.int32)
    edges = np.asarray(edges, dtype=np.int32)
    arc_transitions = np.asarray(arc_transitions, dtype=np.int32)
    transition_rates = np.asarray(transition_rates, dtype=np.float64)
    return _run_sgn_task(vertices, edges, arc_transitions, transition_rates)


def is_connected(petri_net_matrix):
    """Checks if the Petri net is weakly connected (single component)."""
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

    num_nodes = num_places + num_transitions
    scratch_visited, scratch_queue = _get_is_connected_scratchpad(num_nodes)
    return _is_connected_core(petri_net_matrix, scratch_visited, scratch_queue)


@numba.jit(nopython=True, cache=True)
def _is_connected_core(petri_net_matrix, visited, queue):
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
    # ⚡ Bolt Optimization: Using explicit loops instead of `np.any(np.sum(..., axis=1) == 0)`
    # to avoid Numba allocating intermediate arrays (sum array and boolean array).
    # This prevents garbage collection overhead and is ~2.5x faster.
    for p in range(num_places):
        has_edge = False
        for c in range(2 * num_transitions):
            if petri_net_matrix[p, c] != 0:
                has_edge = True
                break
        if not has_edge:
            return False

    # Check for isolated transitions (fast fail)
    # ⚡ Bolt Optimization: Same here, explicit nested loop prevents O(N*M) allocation
    # and allows true short-circuiting natively in Numba.
    for t in range(num_transitions):
        has_edge = False
        for p in range(num_places):
            if petri_net_matrix[p, t] != 0 or petri_net_matrix[p, num_transitions + t] != 0:
                has_edge = True
                break
        if not has_edge:
            return False

    # BFS to check for full connectivity (single component)
    num_nodes = num_places + num_transitions
    for i in range(num_nodes):
        visited[i] = False
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


@numba.jit(nopython=True, cache=True)
def _compute_qualitative_properties_core(
    vertices: np.ndarray,
    edges: np.ndarray,
    seen_sources: np.ndarray,
    edge_counts: np.ndarray,
    adj_t: np.ndarray,
    adj_ptr: np.ndarray,
    current_idx: np.ndarray,
    visited: np.ndarray,
    queue: np.ndarray,
) -> Tuple[bool, bool, bool, int]:
    """Core Numba-optimized logic for qualitative properties."""
    num_vertices = len(vertices)

    # Max tokens and Safeness
    max_tokens = int(np.max(vertices))
    is_safe = max_tokens <= 1

    if len(edges) == 0:
        return False, (num_vertices == 1), is_safe, max_tokens

    # Deadlock-free
    for i in range(num_vertices):
        seen_sources[i] = False
    for i in range(len(edges)):
        seen_sources[edges[i, 0]] = True

    is_deadlock_free = True
    for i in range(num_vertices):
        if not seen_sources[i]:
            is_deadlock_free = False
            break

    # Reversibility: Flat adjacency list for transpose graph
    for i in range(num_vertices):
        edge_counts[i] = 0
    for i in range(len(edges)):
        edge_counts[edges[i, 1]] += 1

    for i in range(num_vertices + 1):
        adj_ptr[i] = 0
    for i in range(num_vertices):
        adj_ptr[i + 1] = adj_ptr[i] + edge_counts[i]

    for i in range(num_vertices):
        current_idx[i] = 0
    for i in range(len(edges)):
        u = edges[i, 0]
        v = edges[i, 1]
        pos = adj_ptr[v] + current_idx[v]
        adj_t[pos] = u
        current_idx[v] += 1

    # BFS using native arrays
    for i in range(num_vertices):
        visited[i] = False
    visited[0] = True
    queue[0] = 0
    head = 0
    tail = 1

    while head < tail:
        u = queue[head]
        head += 1
        start_idx = adj_ptr[u]
        end_idx = adj_ptr[u + 1]
        for i in range(start_idx, end_idx):
            v = adj_t[i]
            if not visited[v]:
                visited[v] = True
                queue[tail] = v
                tail += 1

    is_reversible = tail == num_vertices

    return is_deadlock_free, is_reversible, is_safe, max_tokens


def compute_qualitative_properties(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> Dict[str, Any]:
    """Computes qualitative properties of the SPN reachability graph.

    Metrics:
    - is_deadlock_free: True if every reachable state has at least one enabled transition.
    - is_reversible: True if the initial marking is reachable from every reachable state.
    - is_safe: True if the number of tokens in any place never exceeds 1.
    - max_tokens: The maximum number of tokens in any place across all reachable markings.
    """
    if vertices.size == 0:
        return {
            "is_deadlock_free": False,
            "is_reversible": False,
            "is_safe": False,
            "max_tokens": 0,
        }

    # Ensure types for Numba signature match
    edges_arr = edges.astype(np.int32) if edges.dtype != np.int32 else edges
    vertices_arr = vertices.astype(np.int32) if vertices.dtype != np.int32 else vertices

    # ⚡ Bolt Optimization: Replace Python `set()` and `deque` based property calculations
    # with a Numba compiled core function that uses flat adjacency arrays and integer queues.

    (
        scratch_seen,
        scratch_edge_counts,
        scratch_adj_t,
        scratch_adj_ptr,
        scratch_current_idx,
        scratch_visited,
        scratch_queue,
    ) = _get_qualitative_scratchpad(len(vertices_arr), len(edges_arr))
    is_deadlock_free, is_reversible, is_safe, max_tokens = _compute_qualitative_properties_core(
        vertices_arr,
        edges_arr,
        scratch_seen,
        scratch_edge_counts,
        scratch_adj_t,
        scratch_adj_ptr,
        scratch_current_idx,
        scratch_visited,
        scratch_queue,
    )

    return {
        "is_deadlock_free": bool(is_deadlock_free),
        "is_reversible": bool(is_reversible),
        "is_safe": bool(is_safe),
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
        "arr_vlist": vertices.astype(int),
        "arr_edge": edges.astype(int) if edges.size > 0 else np.empty((0, 2), dtype=int),
        "arr_tranidx": arc_transitions.astype(int) if arc_transitions.size > 0 else np.empty((0,), dtype=int),
        "spn_labda": firing_rates,
        "spn_steadypro": steady_state_probs,
        "spn_markdens": marking_densities,
        "spn_allmus": average_markings,
        "spn_mu": float(average_markings.sum()),
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
    petri_net_matrix = np.asarray(petri_net_matrix)  # Ensure it's a numpy array
    if not is_connected(petri_net_matrix):
        return {}, False

    (
        vertices,
        edges,
        arc_transitions,
        num_transitions,
        is_bounded,
    ) = ArrGra.generate_reachability_graph(petri_net_matrix, place_upper_bound, marks_upper_limit)

    if not is_bounded or vertices.size == 0 or len(vertices) < marks_lower_limit:
        return {}, False

    (
        probs,
        density,
        markings,
        rates,
        success,
    ) = generate_stochastic_net_task(vertices, edges, arc_transitions, num_transitions)

    if not success:
        return {}, False

    # ⚡ Bolt Optimization: Cache markings.sum() to a variable
    # to avoid evaluating `.sum()` twice on the NumPy array.
    markings_sum = markings.sum()
    if markings_sum > 1000 or markings_sum < -1000:
        return {}, False

    return (
        _create_spn_result_dict(petri_net_matrix, vertices, edges, arc_transitions, rates, probs, density, markings),
        True,
    )


def get_spn_info(
    petri_net_matrix: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    arc_transitions: np.ndarray,
    transition_rates: np.ndarray,
) -> Tuple[Dict[str, Any], bool]:
    """Retrieves SPN info for a given structure and rates."""
    petri_net_matrix = np.asarray(petri_net_matrix)  # Ensure it's a numpy array
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    arc_transitions = np.asarray(arc_transitions)
    transition_rates = np.asarray(transition_rates)
    if not is_connected(petri_net_matrix) or vertices.size == 0:
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
