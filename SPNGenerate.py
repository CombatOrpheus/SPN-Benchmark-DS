#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPNGenerate.py
# @Date    : 2020-08-23 (modified for HDF5 output)
# @Author  : mingjian
    描述: Generates Stochastic Petri Net (SPN) datasets and saves them to HDF5.
"""

import argparse
import json
import os

# HDF5 library
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from DataGenerate import DataTransformation  # Contains augment_single_data
from DataGenerate import PetriGenerate as PeGen
# Project-specific modules
from DataGenerate import SPN
from utils import DataUtil as DU


def generate_spn_for_hdf5(config_params):
    """
    Generates a single SPN sample and returns its data as a dictionary.
    This function is intended to be called in parallel.
    The original 'generate_spn' wrote to a JSON; this one returns the dict.

    Args:
        config_params (dict): Configuration dictionary for SPN generation.
        data_idx_unused (int): Index of the data, currently unused in this modified version
                               as the main loop will handle HDF5 group naming.

    Returns:
        dict or None: A dictionary containing the SPN data if generation is successful,
                      None if generation fails in a recoverable way.
    """
    place_upper_bound_cfg = config_params["place_upper_bound"]
    marks_lower_limit_cfg = config_params["marks_lower_limit"]
    marks_upper_limit_cfg = config_params["marks_upper_limit"]
    prune_flag_cfg = config_params["prune_flag"]
    add_token_cfg = config_params["add_token"]
    max_place_number_cfg = config_params["max_place_num"]
    min_place_number_cfg = config_params["min_place_num"]

    spn_generation_finished = False
    results_dict_gen = {}  # Initialize to ensure it's defined

    max_attempts = 100
    attempts = 0
    while not spn_generation_finished and attempts < max_attempts:
        place_number = np.random.randint(min_place_number_cfg, max_place_number_cfg + 1)
        min_trans_offset = -3 if place_number > 3 else 1 - place_number
        max_trans_offset = 0
        if place_number + min_trans_offset <= 0:
            transition_number = 1
        else:
            transition_number = place_number + np.random.randint(min_trans_offset, max_trans_offset + 1)
            if transition_number < 1: transition_number = 1

        petri_matrix = PeGen.generate_random_petri_net(place_number, transition_number)
        if prune_flag_cfg:
            petri_matrix = PeGen.prune_petri_net(petri_matrix)
        if add_token_cfg:
            petri_matrix = PeGen.add_token_to_random_place(petri_matrix)

        results_dict_gen, spn_generation_finished = SPN.filter_stochastic_petri_net(
            petri_matrix, place_upper_bound_cfg, marks_lower_limit_cfg, marks_upper_limit_cfg
        )
        attempts += 1

    if not spn_generation_finished:
        return None

    return results_dict_gen


def augment_single_data_for_hdf5(
        original_sample_dict,
        place_upper_bound_aug,
        marks_lower_limit_aug,
        marks_upper_limit_aug,
        maxtransform_num_aug
):
    """
    Wrapper for DataTransformation.transformation to be used with joblib.

    Args:
        original_sample_dict (dict): The original SPN sample dictionary.
        place_upper_bound_aug (int): Config param.
        marks_lower_limit_aug (int): Config param.
        marks_upper_limit_aug (int): Config param.
        maxtransform_num_aug (int): Max number of transformations to keep.

    Returns:
        list: A list of augmented sample dictionaries.
    """
    if not original_sample_dict or 'petri_net' not in original_sample_dict:
        return []

    petri_net_np = np.array(original_sample_dict["petri_net"], dtype="long")

    all_extended_data = DataTransformation.transformation(
        petri_net_np,
        place_upper_bound_aug,
        marks_lower_limit_aug,
        marks_upper_limit_aug,
    )

    if not all_extended_data:
        return []

    if len(all_extended_data) > maxtransform_num_aug:
        num_to_sample = min(maxtransform_num_aug, len(all_extended_data))
        sample_indices = np.random.choice(
            len(all_extended_data), num_to_sample, replace=False
        )
        transformed_data_list = [all_extended_data[i] for i in sample_indices]
    else:
        transformed_data_list = all_extended_data

    return transformed_data_list


def write_sample_to_hdf5_group(h5_group, sample_dict, compression_filter="gzip", compression_opts_val=4):
    """
    Writes a single sample dictionary to a specified HDF5 group.
    Each key-value pair in the dictionary becomes a dataset in the group.
    Compression is applied only to non-scalar datasets.

    Args:
        h5_group (h5py.Group): The HDF5 group to write into.
        sample_dict (dict): The dictionary containing the sample data.
        compression_filter (str): Compression algorithm (e.g., "gzip", "lzf").
        compression_opts_val (int/tuple): Options for the compression filter.
    """
    for key, value in sample_dict.items():
        try:
            np_value = np.array(value)

            # Check if the value is a scalar
            if np_value.ndim == 0:  # 0-dimensional array is a scalar
                h5_group.create_dataset(key, data=np_value)
            else:  # For arrays, apply compression
                h5_group.create_dataset(
                    key,
                    data=np_value,
                    compression=compression_filter,
                    compression_opts=compression_opts_val
                )
        except TypeError as te:
            print(
                f"Warning: TypeError while converting key '{key}' for sample {h5_group.name}. Data: {value}. Error: {te}")
            print("Attempting to save as string if it's a complex object that can't be array-ized directly.")
            try:
                str_value = str(value)
                h5_group.create_dataset(key, data=str_value)
            except Exception as e_str:
                print(f"Fallback to string failed for key '{key}'. Error: {e_str}")
        except Exception as e:
            print(f"Warning: Could not save key '{key}' for sample {h5_group.name}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Please give a config.json file ",
        default="config/DataConfig/SPNGenerate.json",
    )
    args = parser.parse_args()
    config = DU.load_json(args.config)
    print("Configuration loaded:", config)

    write_data_location = config["write_data_loc"]
    parallel_job = config["parallel_job"]
    place_upper_bound = config["place_upper_bound"]
    marks_lower_limit = config["marks_lower_limit"]
    marks_upper_limit = config["marks_upper_limit"]
    data_number = config["data_num"]
    visual_flag = config["visual_flag"]
    picture_location = config["pic_loc"]
    transformation_flag = config["transformation_flag"]
    maximum_transformation_number = config["maxtransform_num"]

    original_data_location_name = "data_hdf5"
    hdf5_file_path = os.path.join(write_data_location, original_data_location_name, "spn_dataset.hdf5")

    DU.mkdir(os.path.join(write_data_location, original_data_location_name))

    print(f"Output HDF5 file will be: {hdf5_file_path}")

    with h5py.File(hdf5_file_path, 'w') as hf:
        print(f"HDF5 file '{hdf5_file_path}' opened for writing.")
        try:
            config_str = json.dumps(config)
            hf.attrs['generation_config'] = config_str
        except TypeError as e:
            print(f"Warning: Could not serialize full config to JSON for HDF5 attributes. Error: {e}")
            hf.attrs['generation_config_error'] = str(e)

        dataset_h5_group = hf.create_group("dataset_samples")
        hdf5_sample_idx_counter = 0

        print(f"Starting generation of {data_number} initial SPN samples...")

        initial_generated_results = Parallel(n_jobs=parallel_job, backend="loky")(
            delayed(generate_spn_for_hdf5)(config)
            for _ in trange(data_number, desc="Initial Sample Generation")
        )

        initial_samples_list = [res for res in initial_generated_results if res is not None]
        print(f"Successfully generated {len(initial_samples_list)} initial valid samples.")

        if not transformation_flag:
            print(f"Writing {len(initial_samples_list)} initial samples to HDF5 (no transformation)...")
            for sample_dict in tqdm(initial_samples_list, desc="Writing Initial Samples to HDF5"):
                sample_h5_subgroup = dataset_h5_group.create_group(f"sample_{hdf5_sample_idx_counter:07d}")
                write_sample_to_hdf5_group(sample_h5_subgroup, sample_dict)
                hdf5_sample_idx_counter += 1
        else:
            print(f"Starting data augmentation for {len(initial_samples_list)} initial samples...")

            list_of_augmented_sample_lists = Parallel(n_jobs=parallel_job, backend="loky")(
                delayed(augment_single_data_for_hdf5)(
                    sample_dict,
                    place_upper_bound,
                    marks_lower_limit,
                    marks_upper_limit,
                    maximum_transformation_number
                )
                for sample_dict in tqdm(initial_samples_list, desc="Augmenting Samples")
            )

            print("Augmentation complete. Writing augmented samples to HDF5...")
            for augmented_list_from_one_original in tqdm(list_of_augmented_sample_lists,
                                                         desc="Writing Augmented Samples to HDF5"):
                if augmented_list_from_one_original:
                    for augmented_sample_dict in augmented_list_from_one_original:
                        sample_h5_subgroup = dataset_h5_group.create_group(f"sample_{hdf5_sample_idx_counter:07d}")
                        write_sample_to_hdf5_group(sample_h5_subgroup, augmented_sample_dict)
                        hdf5_sample_idx_counter += 1

        hf.attrs['total_samples_written'] = hdf5_sample_idx_counter
        print(f"Total samples written to HDF5: {hdf5_sample_idx_counter}")

    print(f"HDF5 file '{hdf5_file_path}' successfully written and closed.")
