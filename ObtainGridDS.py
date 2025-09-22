"""
This script processes raw data to generate a grid-based dataset for GNN training.
It partitions the data into a grid, samples from each grid cell, and then
packages the data for use with DGL.
"""

import os
import time
import numpy as np
import h5py
from tqdm import tqdm
from utils import DataUtil as DU
from utils import FileWriter as FW
from DataGenerate import DataTransformation as dts


def get_grid_index(value, grid_boundaries):
    """Finds the index of the grid cell for a given value.

    Args:
        value (int): The value to find the grid cell for.
        grid_boundaries (list): A list of boundaries for the grid cells.

    Returns:
        int: The index of the grid cell.
    """
    for i, boundary in enumerate(grid_boundaries):
        if value < boundary:
            return i + 1
    return len(grid_boundaries)


def _initialize_grid(grid_dir, accumulate_data):
    """Initializes the grid structure and configuration."""
    config_path = os.path.join(grid_dir, "config.json")
    if os.path.exists(config_path) and accumulate_data:
        grid_config = DU.load_json_file(config_path)
    else:
        grid_config = {
            "row_p": [5 + 2 * (i + 1) for i in range(5)],
            "col_m": [4 + 4 * (i + 1) for i in range(10)],
            "json_count": np.zeros((5, 10), dtype=int).tolist(),
        }

    for i in range(len(grid_config["row_p"])):
        for j in range(len(grid_config["col_m"])):
            DU.create_directory(os.path.join(grid_dir, f"p{i+1}", f"m{j+1}"))

    return grid_config


def partition_data_into_grid(grid_dir, accumulate_data, raw_data_path):
    """Partitions raw data into a grid structure.

    Args:
        grid_dir (str): The directory to store the grid data.
        accumulate_data (bool): Whether to accumulate data or start fresh.
        raw_data_path (str): The path to the raw data JSON file.
    """
    grid_config = _initialize_grid(grid_dir, accumulate_data)
    row_p = grid_config["row_p"]
    col_m = grid_config["col_m"]
    dir_counts = np.array(grid_config["json_count"])

    all_data = DU.load_json_file(raw_data_path)

    for data in all_data.values():
        p_idx = get_grid_index(len(data["petri_net"]), row_p)
        m_idx = get_grid_index(len(data["arr_vlist"]), col_m)
        dir_counts[p_idx - 1, m_idx - 1] += 1

        save_path = os.path.join(
            grid_dir,
            f"p{p_idx}",
            f"m{m_idx}",
            f"data{int(dir_counts[p_idx-1, m_idx-1])}.json",
        )
        DU.save_data_to_json_file(save_path, data)

    grid_config["json_count"] = dir_counts.tolist()
    DU.save_data_to_json_file(os.path.join(grid_dir, "config.json"), grid_config)


def sample_and_transform_data(config):
    """Samples data from the grid and applies transformations."""
    grid_data_loc = os.path.join(config["temporary_grid_location"], "p%s", "m%s")
    all_data = []
    for i in range(config["places_upper_limit"]):
        for j in range(config["markings_upper_limit"]):
            sampled_list = DU.sample_json_files_from_directory(
                config["samples_per_grid"], grid_data_loc % (i + 1, j + 1)
            )
            all_data.extend(sampled_list)

    transformed_data = []
    for data in all_data:
        new_data = dts.generate_lambda_variations(data, config["lambda_variations_per_sample"])
        transformed_data.extend(new_data)

    return transformed_data


def package_dataset(config, data):
    """Saves the processed data into the specified format (HDF5 or JSON-L)."""
    save_dir = config["output_grid_location"]
    output_format = config.get("output_format", "hdf5")
    output_file = config.get("output_file", f"grid_dataset.{output_format}")
    output_path = os.path.join(save_dir, output_file)

    DU.create_directory(save_dir)

    if output_format == "hdf5":
        with h5py.File(output_path, "w") as hf:
            hf.attrs["generation_config"] = json.dumps(config, cls=FW.NumpyEncoder)
            dataset_group = hf.create_group("dataset_samples")

            print(f"Writing {len(data)} samples to HDF5...")
            for i, sample in enumerate(tqdm(data, desc="Writing to HDF5")):
                sample_group = dataset_group.create_group(f"sample_{i:07d}")
                FW.write_to_hdf5(sample_group, sample)

            hf.attrs["total_samples_written"] = len(data)
        print(f"HDF5 file '{output_path}' created successfully.")

    elif output_format == "jsonl":
        with open(output_path, "w") as f:
            f.write(json.dumps(config, cls=FW.NumpyEncoder) + "\n")

            print(f"Writing {len(data)} samples to JSONL...")
            for sample in tqdm(data, desc="Writing to JSONL"):
                FW.write_to_jsonl(f, sample)
        print(f"JSONL file '{output_path}' created successfully.")


import argparse
import json


def main():
    """Main function to generate the grid dataset."""
    parser = argparse.ArgumentParser(
        description="Process raw data to generate a grid-based dataset for GNN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/DataConfig/PartitionGrid.toml",
        help="Path to config TOML file.",
    )
    args = parser.parse_args()
    config = DU.load_toml_file(args.config)

    partition_data_into_grid(
        config["temporary_grid_location"],
        config["accumulation_data"],
        config["raw_data_location"],
    )

    processed_data = sample_and_transform_data(config)

    package_dataset(config, processed_data)


if __name__ == "__main__":
    main()
