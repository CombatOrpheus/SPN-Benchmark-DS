import time
import numpy as np
from scipy.sparse import csc_array, coo_array
import numba

@numba.jit(nopython=True, cache=True)
def _compute_state_equation_numba_old(num_vertices, edges, arc_transitions, lambda_values):
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

def compute_state_equation_old(num_vertices, edges_arr, arc_transitions, lambda_values):
    state_matrix_np = _compute_state_equation_numba_old(
        num_vertices, edges_arr, arc_transitions, lambda_values
    )
    return csc_array(state_matrix_np)

@numba.jit(nopython=True, cache=True)
def _compute_state_equation_coo(num_vertices, edges, arc_transitions, lambda_values):
    num_edges = len(edges)
    num_entries = 2 * num_edges + num_vertices

    rows = np.zeros(num_entries, dtype=np.int32)
    cols = np.zeros(num_entries, dtype=np.int32)
    data = np.zeros(num_entries, dtype=np.float64)

    idx = 0
    for i in range(num_edges):
        edge = edges[i]
        trans_idx = arc_transitions[i]
        src_idx, dest_idx = edge[0], edge[1]
        rate = lambda_values[trans_idx]

        rows[idx] = src_idx
        cols[idx] = src_idx
        data[idx] = -rate
        idx += 1

        rows[idx] = dest_idx
        cols[idx] = src_idx
        data[idx] = rate
        idx += 1

    for i in range(num_vertices):
        rows[idx] = num_vertices
        cols[idx] = i
        data[idx] = 1.0
        idx += 1

    return data, rows, cols

def compute_state_equation_new(num_vertices, edges_arr, arc_transitions, lambda_values):
    data, rows, cols = _compute_state_equation_coo(
        num_vertices, edges_arr, arc_transitions, lambda_values
    )
    return coo_array((data, (rows, cols)), shape=(num_vertices + 1, num_vertices)).tocsc()

# Generate some dummy data
num_vertices = 5000
num_edges = 15000

edges = np.random.randint(0, num_vertices, size=(num_edges, 2), dtype=np.int32)
arc_transitions = np.random.randint(0, 10, size=num_edges, dtype=np.int32)
lambda_values = np.random.rand(10)

# warmup
old_res = compute_state_equation_old(num_vertices, edges, arc_transitions, lambda_values)
new_res = compute_state_equation_new(num_vertices, edges, arc_transitions, lambda_values)
assert np.allclose(old_res.toarray(), new_res.toarray())

# benchmark old
start = time.time()
for _ in range(100):
    compute_state_equation_old(num_vertices, edges, arc_transitions, lambda_values)
print(f"Old time: {time.time() - start:.4f} seconds")

# benchmark new
start = time.time()
for _ in range(100):
    compute_state_equation_new(num_vertices, edges, arc_transitions, lambda_values)
print(f"New time: {time.time() - start:.4f} seconds")
