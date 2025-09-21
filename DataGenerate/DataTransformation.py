"""
This module provides functions for augmenting Petri net data by generating
variations of a given Petri net structure and its firing rates.
"""

import numpy as np
from joblib import Parallel, delayed
from DataGenerate import SPN


def _generate_candidate_matrices(base_petri_matrix):
    """Generates candidate Petri net matrices by applying various transformations."""
    candidate_matrices = []
    num_places, num_cols = base_petri_matrix.shape
    num_transitions = (num_cols - 1) // 2

    # Delete an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 0
                candidate_matrices.append(modified_matrix)

    # Add an edge
    for r in range(num_places):
        for c in range(num_cols - 1):
            if base_petri_matrix[r, c] == 0:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, c] = 1
                candidate_matrices.append(modified_matrix)

    # Add a token
    for r in range(num_places):
        modified_matrix = base_petri_matrix.copy()
        modified_matrix[r, -1] += 1
        candidate_matrices.append(modified_matrix)

    # Delete a token
    if np.sum(base_petri_matrix[:, -1]) > 1:
        for r in range(num_places):
            if base_petri_matrix[r, -1] >= 1:
                modified_matrix = base_petri_matrix.copy()
                modified_matrix[r, -1] -= 1
                candidate_matrices.append(modified_matrix)

    # Add a place
    if num_transitions > 0:
        new_place_row = np.zeros((1, num_cols), dtype=int)
        t_idx_to_connect = np.random.randint(0, num_transitions * 2)
        new_place_row[0, t_idx_to_connect] = 1
        modified_matrix = np.vstack([base_petri_matrix, new_place_row])
        candidate_matrices.append(modified_matrix)

    return candidate_matrices


def _generate_rate_variations(base_variation, num_variations):
    """Generates variations of firing rates for a given Petri net structure."""
    rate_variations = []
    p_net = base_variation["petri_net"]
    num_trans = (p_net.shape[1] - 1) // 2
    if num_trans == 0:
        return []

    for _ in range(num_variations):
        new_rates = np.random.randint(1, 11, size=num_trans).astype(float)
        s_probs, m_dens, avg_marks, success = SPN.generate_stochastic_net_task_with_rates(
            [v for v in base_variation["arr_vlist"]],
            base_variation["arr_edge"].tolist(),
            base_variation["arr_tranidx"].tolist(),
            new_rates,
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
                "spn_mu": np.sum(avg_marks),
            }
            rate_variations.append(new_result)
    return rate_variations


def generate_petri_net_variations(
    petri_matrix,
    place_upper_bound,
    marks_lower_limit,
    marks_upper_limit,
    parallel_job_count=1,
    num_rate_variations_per_structure=5,
    max_candidates_per_structure=50,
):
    """Generates variations of a Petri net to augment the dataset.

    Args:
        petri_matrix (numpy.ndarray): The base Petri net matrix.
        place_upper_bound (int): The upper bound for tokens in any single place.
        marks_lower_limit (int): The lower limit for the number of markings.
        marks_upper_limit (int): The upper limit for the number of markings.
        parallel_job_count (int, optional): The number of parallel jobs to run. Defaults to 1.
        num_rate_variations_per_structure (int, optional): The number of firing rate
            variations to generate for each valid structure. Defaults to 5.
        max_candidates_per_structure (int, optional): The maximum number of candidate
            structures to consider. Defaults to 50.

    Returns:
        list: A list of dictionaries, each representing an augmented Petri net.
    """
    base_petri_matrix = np.array(petri_matrix)
    candidate_matrices = _generate_candidate_matrices(base_petri_matrix)
    if len(candidate_matrices) > max_candidates_per_structure:
        candidate_matrices = candidate_matrices[:max_candidates_per_structure]

    results = Parallel(n_jobs=parallel_job_count)(
        delayed(SPN.filter_spn)(
            matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
        )
        for matrix in candidate_matrices
    )
    structural_variations = [res for res, success in results if success]

    all_augmented_data = []
    all_augmented_data.extend(structural_variations)

    for base_variation in structural_variations:
        rate_variations = _generate_rate_variations(base_variation, num_rate_variations_per_structure)
        all_augmented_data.extend(rate_variations)

    return all_augmented_data


def generate_lambda_variations(petri_dict, num_lambda_variations):
    """Generates variations of lambda values for a given Petri net.

    Args:
        petri_dict (dict): A dictionary representing the Petri net.
        num_lambda_variations (int): The number of lambda variations to generate.

    Returns:
        list: A list of dictionaries, each representing a Petri net with new lambda values.
    """
    lambda_variations = []
    petri_net = petri_dict["petri_net"]
    num_transitions = (len(petri_net[0]) - 1) // 2

    while len(lambda_variations) < num_lambda_variations:
        lambda_values = np.random.randint(1, 11, size=num_transitions)
        results_dict, success = SPN.get_stochastic_petri_net(
            petri_net,
            petri_dict["arr_vlist"],
            petri_dict["arr_edge"],
            petri_dict["arr_tranidx"],
            lambda_values,
        )
        if success:
            lambda_variations.append(results_dict)

    return lambda_variations
