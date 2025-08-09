#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : ObtainGridDS.py
@Date    : 2020-09-21
@Author  : mingjian

This script processes a raw dataset of Stochastic Petri Nets (SPNs),
partitions them into a grid based on place and marking counts, applies
transformations, and packages the final data into a DGL-compatible dataset.
"""

import os
import pickle
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from GNNs.datasets.NetLearningDatasetDGL import NetLearningDatasetDGL
from utils import DataUtil as DU
from DataGenerate import DataTransformation as dts


def find_grid_index(value: int, bins: List[int]) -> int:
    """
    Finds the 1-based index of the bin that a value falls into.

    This function determines which "bucket" or "bin" a given value belongs to.
    For example, if bins are `[10, 20, 30]`, a value of 5 would be in bin 1,
    15 in bin 2, and 25 in bin 3. A value of 35 would be in bin 4.

    Note: A more efficient way to do this is with `np.searchsorted`.

    Args:
        value: The value to categorize.
        bins: A sorted list of upper bounds for the bins.

    Returns:
        The 1-based index of the bin.
    """
    for i, upper_bound in enumerate(bins):
        if value < upper_bound:
            return i + 1
    return len(bins)


def partition_data_into_grid(
    temp_grid_dir: str, accumulate_data: bool, raw_data_path: str
):
    """
    Partitions raw SPN data into a grid structure based on the number of places and markings.

    Args:
        temp_grid_dir: The directory to save the partitioned grid data.
        accumulate_data: If True, adds to existing grid data.
        raw_data_path: The path to the raw JSON data file.
    """
    # Define the grid dimensions
    place_bins = [5 + 2 * (i + 1) for i in range(5)]
    marking_bins = [4 + 4 * (i + 1) for i in range(10)]

    # Create the grid directories
    for i in range(len(place_bins)):
        for j in range(len(marking_bins)):
            DU.mkdir(os.path.join(temp_grid_dir, f"p{i+1}", f"m{j+1}"))

    grid_config_path = os.path.join(temp_grid_dir, "config.json")
    if os.path.exists(grid_config_path) and accumulate_data:
        grid_config = DU.load_json(grid_config_path)
        counts = np.array(grid_config["json_count"])
    else:
        counts = np.zeros((len(place_bins), len(marking_bins)), dtype=int)

    all_data = DU.load_json(raw_data_path)
    print(f"Loaded {len(all_data)} raw data samples.")

    # Partition each data sample into the grid
    for data_sample in all_data.values():
        num_places = len(data_sample["petri_net"])
        num_markings = len(data_sample["arr_vlist"])

        p_idx = find_grid_index(num_places, place_bins)
        m_idx = find_grid_index(num_markings, marking_bins)

        counts[p_idx - 1, m_idx - 1] += 1
        filepath = os.path.join(
            temp_grid_dir,
            f"p{p_idx}",
            f"m{m_idx}",
            f"data{int(counts[p_idx-1, m_idx-1])}.json",
        )
        DU.save_data_to_json(filepath, data_sample)

    # Save the updated grid configuration
    grid_config = {
        "row_p": place_bins,
        "col_m": marking_bins,
        "json_count": counts.tolist(),
    }
    DU.save_data_to_json(grid_config_path, grid_config)
    print("Finished partitioning data into grid.")


def sample_and_transform_from_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Samples data from the grid, applies lambda transformations, and returns the result.
    """
    temp_grid_dir = config["tmp_grid_loc"]
    p_limit = config["p_upper_limit"]
    m_limit = config["m_upper_limit"]
    samples_per_grid = config["each_grid_num"]
    lambda_transformations_per_sample = config["labda_num"]

    all_sampled_data = []
    grid_cell_path_template = os.path.join(temp_grid_dir, "p%s", "m%s")

    for i in range(p_limit):
        for j in range(m_limit):
            grid_cell_path = grid_cell_path_template % (i + 1, j + 1)
            sampled_in_cell = DU.sample_dir_json(samples_per_grid, grid_cell_path)
            all_sampled_data.extend(sampled_in_cell)

    print(f"Sampled {len(all_sampled_data)} total data points from the grid.")

    transformed_data = []
    for data in all_sampled_data:
        # Note: The original script used a function named `labda_transformation`.
        # Assuming `dts.generate_samples_with_new_rates` is the intended replacement.
        new_samples = dts.generate_samples_with_new_rates(
            data, lambda_transformations_per_sample
        )
        transformed_data.extend(new_samples)

    print(f"Generated {len(transformed_data)} samples after lambda transformation.")
    return transformed_data


def package_final_dataset(final_data: List[Dict[str, Any]], save_dir: str):
    """
    Packages the final data into the required directory structure and file formats.
    """
    # Save the final data as a single JSON file
    DU.mkdir(os.path.join(save_dir, "ori_data"))
    final_data_dict = DU.gen_dict(final_data)
    DU.save_data_to_json(
        os.path.join(save_dir, "ori_data", "all_data.json"), final_data_dict
    )

    # Partition into train/test sets and create preprocessed data
    DU.partition_datasets(save_dir, batch_size=16, test_ratio=0.2)

    # Create and save the DGL dataset
    DU.mkdir(os.path.join(save_dir, "package_data"))
    dataset = NetLearningDatasetDGL(os.path.join(save_dir, "preprocessd_data"))

    print("Packaging final dataset into a pickle file...")
    start_time = time.time()
    with open(os.path.join(save_dir, "package_data", "dataset.pkl"), "wb") as f:
        pickle.dump([dataset.train, dataset.test], f)
    print(f"Dataset packaging took: {time.time() - start_time:.2f} seconds.")


def main():
    """
    Main workflow to generate the grid dataset.
    1. Partitions raw data into a grid.
    2. Samples from the grid and applies transformations.
    3. Packages the final dataset for use in training.
    """
    config = DU.load_json("config/DataConfig/PartitionGrid.json")

    # Step 1: Partition the raw data into a grid structure.
    partition_data_into_grid(
        config["tmp_grid_loc"],
        config["accumulation_data"],
        config["raw_data_loc"],
    )

    # Step 2: Sample from the grid and apply transformations.
    final_data_to_package = sample_and_transform_from_grid(config)

    # Step 3: Package the final dataset.
    package_final_dataset(final_data_to_package, config["save_grid_loc"])


if __name__ == "__main__":
    main()
