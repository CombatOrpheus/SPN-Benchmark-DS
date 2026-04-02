import numpy as np
import time
import numba

@numba.jit(nopython=True, cache=True)
def delete_excess_edges_opt(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]
    num_cols = petri_matrix.shape[1] - 1

    for i in range(num_places):
        # count instead of sum
        edge_indices = np.empty(num_cols, dtype=np.int64)
        count = 0
        for j in range(num_cols):
            if petri_matrix[i, j] == 1:
                edge_indices[count] = j
                count += 1

        if count >= 3:
            indices_to_remove = np.random.permutation(edge_indices[:count])[: count - 2]
            for j in range(len(indices_to_remove)):
                petri_matrix[i, indices_to_remove[j]] = 0

    for i in range(2 * num_transitions):
        edge_indices = np.empty(num_places, dtype=np.int64)
        count = 0
        for j in range(num_places):
            if petri_matrix[j, i] == 1:
                edge_indices[count] = j
                count += 1

        if count >= 3:
            indices_to_remove = np.random.permutation(edge_indices[:count])[: count - 2]
            for j in range(len(indices_to_remove)):
                petri_matrix[indices_to_remove[j], i] = 0

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def add_missing_connections_opt(petri_matrix, num_transitions):
    num_places = petri_matrix.shape[0]

    zero_sum_cols = np.empty(2 * num_transitions, dtype=np.int64)
    zero_sum_cols_count = 0
    for j in range(2 * num_transitions):
        has_edge = False
        for i in range(num_places):
            if petri_matrix[i, j] != 0:
                has_edge = True
                break
        if not has_edge:
            zero_sum_cols[zero_sum_cols_count] = j
            zero_sum_cols_count += 1

    if zero_sum_cols_count > 0:
        random_rows = np.random.randint(0, num_places, size=zero_sum_cols_count)
        for i in range(zero_sum_cols_count):
            petri_matrix[random_rows[i], zero_sum_cols[i]] = 1

    rows_with_zero_pre_sum = np.empty(num_places, dtype=np.int64)
    rows_with_zero_pre_sum_count = 0
    for i in range(num_places):
        has_edge = False
        for j in range(num_transitions):
            if petri_matrix[i, j] != 0:
                has_edge = True
                break
        if not has_edge:
            rows_with_zero_pre_sum[rows_with_zero_pre_sum_count] = i
            rows_with_zero_pre_sum_count += 1

    if rows_with_zero_pre_sum_count > 0:
        random_cols_pre = np.random.randint(0, num_transitions, size=rows_with_zero_pre_sum_count)
        for i in range(rows_with_zero_pre_sum_count):
            petri_matrix[rows_with_zero_pre_sum[i], random_cols_pre[i]] = 1

    rows_with_zero_post_sum = np.empty(num_places, dtype=np.int64)
    rows_with_zero_post_sum_count = 0
    for i in range(num_places):
        has_edge = False
        for j in range(num_transitions):
            if petri_matrix[i, num_transitions + j] != 0:
                has_edge = True
                break
        if not has_edge:
            rows_with_zero_post_sum[rows_with_zero_post_sum_count] = i
            rows_with_zero_post_sum_count += 1

    if rows_with_zero_post_sum_count > 0:
        random_cols_post = np.random.randint(0, num_transitions, size=rows_with_zero_post_sum_count)
        for i in range(rows_with_zero_post_sum_count):
            petri_matrix[rows_with_zero_post_sum[i], random_cols_post[i] + num_transitions] = 1

    return petri_matrix

np.random.seed(42)
petri_matrices = []
for _ in range(1000):
    petri_matrix = np.random.randint(0, 2, size=(50, 101), dtype=np.int32)
    petri_matrices.append(petri_matrix)

# Warmup
delete_excess_edges_opt(petri_matrices[0], 50)
add_missing_connections_opt(petri_matrices[0], 50)

start = time.time()
for m in petri_matrices:
    delete_excess_edges_opt(m, 50)
end = time.time()
print(f"delete_excess_edges_opt Time: {end - start:.4f}s")

start = time.time()
for m in petri_matrices:
    add_missing_connections_opt(m, 50)
end = time.time()
print(f"add_missing_connections_opt Time: {end - start:.4f}s")
