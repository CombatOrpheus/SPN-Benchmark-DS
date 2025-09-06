#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : DataTransformation.py
# @Date    : 2020-08-23
# @Author  : mingjian
    描述
"""

import numpy as np
from joblib import Parallel, delayed
from DataGenerate import SPN


def transformation(
    petri_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit, parallel_job_count=1
):
    """
    Generates variations of a given Petri net to augment the dataset.
    This function has been refactored to be more efficient by avoiding
    redundant reachability graph calculations and by parallelizing the
    validation of structural variations.
    """
    base_petri_matrix = np.array(petri_matrix)

    # --- Step 1: Generate all candidate Petri net structures ---
    candidate_matrices = []

    num_places, num_cols = base_petri_matrix.shape
    num_transitions = (num_cols - 1) // 2

    # Transformation 1: Delete an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 0
                candidate_matrices.append(modified_matrix)

    # Transformation 2: Add an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 0:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 1
                candidate_matrices.append(modified_matrix)

    # Transformation 3: Add a token
    for r in range(num_places):
        modified_matrix = base_petri_matrix.copy()
        modified_matrix[r, num_cols - 1] += 1
        candidate_matrices.append(modified_matrix)

    # Transformation 4: Delete a token
    if np.sum(base_petri_matrix[:, -1]) > 1:
        for r in range(num_places):
            if base_petri_matrix[r, num_cols - 1] >= 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, num_cols - 1] -= 1
                candidate_matrices.append(modified_matrix)

    # Transformation 5: Add a place
    if num_transitions > 0:
        new_place_row = np.zeros((1, num_cols), dtype=int)
        t_idx_to_connect = np.random.randint(0, num_transitions * 2)
        new_place_row[0, t_idx_to_connect] = 1
        modified_matrix = np.vstack([base_petri_matrix, new_place_row])
        candidate_matrices.append(modified_matrix)

    # --- Step 2: Filter the candidates in parallel ---
    results = Parallel(n_jobs=parallel_job_count)(
        delayed(SPN.filter_stochastic_petri_net)(
            matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
        ) for matrix in candidate_matrices
    )

    # Collect successful transformations
    structural_variations = [res for res, success in results if success]

    # --- Step 3: Generate more samples by varying firing rates for each valid structure ---
    all_augmented_data = []
    all_augmented_data.extend(structural_variations)

    num_rate_variations_per_structure = 5

    for base_variation in structural_variations:
        p_net = base_variation["petri_net"]
        v_list = [v for v in base_variation["arr_vlist"]]
        e_list = base_variation["arr_edge"].tolist()
        t_indices = base_variation["arr_tranidx"].tolist()
        num_trans = (p_net.shape[1] - 1) // 2

        if num_trans == 0: continue

        # This part can also be parallelized, but the overhead might be larger than the benefit
        # for small numbers of variations. Keeping it sequential for now.
        for _ in range(num_rate_variations_per_structure):
            new_rates = np.random.randint(1, 11, size=num_trans).astype(float)

            s_probs, m_dens, avg_marks, success = SPN.generate_stochastic_graphical_net_task_with_given_rates(
                v_list, e_list, t_indices, new_rates
            )

            if success:
                new_result = {
                    "petri_net": p_net,
                    "arr_vlist": base_variation["arr_vlist"],
                    "arr_edge": base_variation["arr_edge"],
                    "arr_tranidx": base_variation["arr_tranidx"],
                    "spn_labda": new_rates,
                    "spn_steadypro": s_probs,
                    "spn_markdens": m_dens,
                    "spn_allmus": avg_marks,
                    "spn_mu": np.sum(avg_marks)
                }
                all_augmented_data.append(new_result)

    return all_augmented_data


def labda_transformation(petri_dict, labda_num):
    all_labda_list = []
    # petri_matrix = np.array(petri_matrix)
    petri_net = petri_dict["petri_net"]
    arr_vlist = petri_dict["arr_vlist"]
    arr_edge = petri_dict["arr_edge"]
    arr_tranidx = petri_dict["arr_tranidx"]
    tran_num = (len(petri_net[0]) - 1) // 2
    while len(all_labda_list) < labda_num:
        labda = np.random.randint(1, 11, size=tran_num)
        results_dict, finish = SPN.get_stochastic_petri_net(
            petri_net, arr_vlist, arr_edge, arr_tranidx, labda
        )
        if finish:
            all_labda_list.append(results_dict)
    return all_labda_list
