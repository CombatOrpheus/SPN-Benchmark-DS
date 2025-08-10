#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : DataTransformation.py
# @Date    : 2020-08-23
# @Author  : mingjian
    描述
"""

import numpy as np
from DataGenerate import SPN


def transformation(
    petri_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
):
    """
    Generates variations of a given Petri net to augment the dataset.
    This function has been refactored to be more efficient by avoiding
    redundant reachability graph calculations.
    """
    base_petri_matrix = np.array(petri_matrix)

    # --- Step 1: Generate a set of unique, valid Petri net structures ---
    # This list will hold the results of successful structural transformations.
    # Each item is a dictionary containing the full SPN solution.
    structural_variations = []

    num_places, num_cols = base_petri_matrix.shape
    num_transitions = (num_cols - 1) // 2

    # Transformation 1: Delete an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 0
                res, success = SPN.filter_stochastic_petri_net(
                    modified_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
                )
                if success:
                    structural_variations.append(res)

    # Transformation 2: Add an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 0:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 1
                res, success = SPN.filter_stochastic_petri_net(
                    modified_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
                )
                if success:
                    structural_variations.append(res)

    # Transformation 3: Add a token
    for r in range(num_places):
        modified_matrix = base_petri_matrix.copy()
        modified_matrix[r, num_cols - 1] += 1
        res, success = SPN.filter_stochastic_petri_net(
            modified_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
        )
        if success:
            structural_variations.append(res)

    # Transformation 4: Delete a token
    if np.sum(base_petri_matrix[:, -1]) > 1:
        for r in range(num_places):
            if base_petri_matrix[r, num_cols - 1] >= 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, num_cols - 1] -= 1
                res, success = SPN.filter_stochastic_petri_net(
                    modified_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
                )
                if success:
                    structural_variations.append(res)

    # Transformation 5: Add a place (This is a complex transformation, keeping it simple)
    # The original implementation was very complex and likely inefficient.
    # A simpler version could be to add a place and connect it to one existing transition.
    if num_transitions > 0:
        new_place_row = np.zeros((1, num_cols), dtype=int)
        # Connect the new place to a random transition
        t_idx_to_connect = np.random.randint(0, num_transitions * 2)
        new_place_row[0, t_idx_to_connect] = 1

        modified_matrix = np.vstack([base_petri_matrix, new_place_row])
        res, success = SPN.filter_stochastic_petri_net(
            modified_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
        )
        if success:
            structural_variations.append(res)


    # --- Step 2: Generate more samples by varying firing rates for each valid structure ---
    # This is much more efficient as it reuses the expensive-to-compute reachability graph.

    all_augmented_data = []
    # Add the initial successful transformations to the final list
    all_augmented_data.extend(structural_variations)

    # For each valid structure we found, generate a few more samples with different firing rates.
    num_rate_variations_per_structure = 5  # Can be adjusted

    for base_variation in structural_variations:
        # Extract the reachability graph and structure info
        p_net = base_variation["petri_net"]
        # The v_list needs to be a list of arrays for the solver function
        v_list = [v for v in base_variation["arr_vlist"]]
        e_list = base_variation["arr_edge"].tolist()
        t_indices = base_variation["arr_tranidx"].tolist()
        num_trans = (p_net.shape[1] - 1) // 2

        if num_trans == 0: continue

        for _ in range(num_rate_variations_per_structure):
            # Generate new random firing rates
            new_rates = np.random.randint(1, 11, size=num_trans).astype(float)

            # Re-solve the SPN with the new rates, reusing the graph
            s_probs, m_dens, avg_marks, success = SPN.generate_stochastic_graphical_net_task_with_given_rates(
                v_list, e_list, t_indices, new_rates
            )

            if success:
                # If successful, assemble the new results dictionary
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
