#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : DataTransformation.py
@Date    : 2020-08-23
@Author  : mingjian

This module provides functions for data augmentation of Stochastic Petri Nets (SPNs).
It can generate new, valid SPN samples from an existing one by applying various
structural transformations, such as adding or removing arcs and places, or by
modifying the initial marking and firing rates.
"""

from typing import List, Dict, Any

import numpy as np
from DataGenerate import SPN


def _try_add_remove_arcs(
    petri_matrix: np.ndarray, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generates new SPNs by adding or removing one arc at a time."""
    augmented_data = []
    num_places, num_cols = petri_matrix.shape

    for r in range(num_places):
        for c in range(num_cols - 1):  # Exclude marking column
            # Try removing an existing arc
            if petri_matrix[r, c] == 1:
                modified_matrix = petri_matrix.copy()
                modified_matrix[r, c] = 0
                res, success = SPN.filter_stochastic_petri_net(modified_matrix, **config)
                if success:
                    augmented_data.append(res)

            # Try adding a new arc
            else:
                modified_matrix = petri_matrix.copy()
                modified_matrix[r, c] = 1
                res, success = SPN.filter_stochastic_petri_net(modified_matrix, **config)
                if success:
                    augmented_data.append(res)
    return augmented_data


def _try_add_remove_tokens(
    petri_matrix: np.ndarray, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generates new SPNs by adding or removing one token at a time."""
    augmented_data = []
    num_places = petri_matrix.shape[0]

    for i in range(num_places):
        # Try adding a token
        add_matrix = petri_matrix.copy()
        add_matrix[i, -1] += 1
        res_add, success_add = SPN.filter_stochastic_petri_net(add_matrix, **config)
        if success_add:
            augmented_data.append(res_add)

        # Try removing a token, if possible
        if petri_matrix[i, -1] > 0 and np.sum(petri_matrix[:, -1]) > 1:
            del_matrix = petri_matrix.copy()
            del_matrix[i, -1] -= 1
            res_del, success_del = SPN.filter_stochastic_petri_net(del_matrix, **config)
            if success_del:
                augmented_data.append(res_del)
    return augmented_data


def _try_add_place(
    petri_matrix: np.ndarray, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generates new SPNs by adding a new place with two connections."""
    augmented_data = []
    num_cols = petri_matrix.shape[1]

    for i in range(num_cols - 2):
        for k in range(i + 1, num_cols - 1):
            new_place_row = np.zeros((1, num_cols))
            modified_matrix = np.row_stack((petri_matrix, new_place_row))
            modified_matrix[-1, i] = 1
            modified_matrix[-1, k] = 1
            res, success = SPN.filter_stochastic_petri_net(modified_matrix, **config)
            if success:
                augmented_data.append(res)
    return augmented_data


def transformation(
    petri_matrix: np.ndarray,
    place_upper_bound: int,
    marks_lower_limit: int,
    marks_upper_limit: int,
    target_augmentation_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Applies a set of structural transformations to a Petri net to generate a
    dataset of augmented, valid SPN samples.

    Args:
        petri_matrix: The base Petri net matrix to transform.
        place_upper_bound: Max tokens per place for filtering.
        marks_lower_limit: Min reachable markings for filtering.
        marks_upper_limit: Max reachable markings for filtering.
        target_augmentation_size: The desired number of augmented samples.

    Returns:
        A list of dictionaries, where each dictionary represents a valid,
        augmented SPN.
    """
    config = {
        "place_upper_bound": place_upper_bound,
        "marks_lower_limit": marks_lower_limit,
        "marks_upper_limit": marks_upper_limit,
    }

    # Generate a base set of augmentations from structural changes
    base_augmentations = []
    base_augmentations.extend(_try_add_remove_arcs(petri_matrix, config))
    base_augmentations.extend(_try_add_remove_tokens(petri_matrix, config))
    base_augmentations.extend(_try_add_place(petri_matrix, config))

    # From the valid augmentations, generate more samples by varying firing rates
    final_augmented_list = []
    if not base_augmentations: # If no structural changes worked, use the original
        valid_original, success = SPN.filter_stochastic_petri_net(petri_matrix, **config)
        if success:
            base_augmentations = [valid_original]

    # Generate new samples by re-running the solver with random firing rates
    # until the target size is reached.
    while len(final_augmented_list) < target_augmentation_size and base_augmentations:
        # Pick a random valid SPN from our augmented set
        random_spn_dict = np.random.choice(base_augmentations)

        # Re-solve it with new random firing rates
        res, success = SPN.filter_stochastic_petri_net(random_spn_dict['petri_net'], **config)
        if success:
            final_augmented_list.append(res)

    return final_augmented_list


def generate_samples_with_new_rates(
    petri_dict: Dict[str, Any],
    num_samples: int
) -> List[Dict[str, Any]]:
    """
    Generates a number of new SPN samples from an existing SPN structure
    by solving it with different, randomly generated firing rates.

    Args:
        petri_dict: A dictionary representing a valid SPN.
        num_samples: The number of new samples to generate.

    Returns:
        A list of dictionaries, each representing a new SPN sample.
    """
    newly_generated_samples = []

    # Extract structural info from the input dictionary
    petri_net = petri_dict.get("petri_net")
    if petri_net is None:
        return []

    arr_vlist = petri_dict.get("arr_vlist")
    arr_edge = petri_dict.get("arr_edge")
    arr_tranidx = petri_dict.get("arr_tranidx")
    num_transitions = (petri_net.shape[1] - 1) // 2

    while len(newly_generated_samples) < num_samples:
        # Generate new random firing rates
        new_rates = np.random.randint(1, 11, size=num_transitions)

        # Solve the SPN with the new rates
        results_dict, success = SPN.get_stochastic_petri_net(
            petri_net, arr_vlist, arr_edge, arr_tranidx, new_rates
        )
        if success:
            newly_generated_samples.append(results_dict)

    return newly_generated_samples
