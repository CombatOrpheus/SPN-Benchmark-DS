#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : SPNGenerate.py
@Date    : 2020-08-23 (modified for HDF5 output)
@Author  : mingjian

This script generates Stochastic Petri Net (SPN) datasets and saves them to HDF5.
It can generate a set of initial SPNs and optionally apply data augmentation
to create a larger, more diverse dataset.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from DataGenerate import DataTransformation, PetriGenerate as PeGen, SPN
from utils import DataUtil as DU


def generate_single_spn(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates a single valid Stochastic Petri Net sample.

    This function is designed to be called in parallel. It attempts to generate
    a valid SPN within a fixed number of attempts.

    Args:
        config: A dictionary containing configuration parameters for generation.

    Returns:
        A dictionary containing the SPN data if successful, otherwise None.
    """
    max_attempts = 100
    for _ in range(max_attempts):
        place_num = np.random.randint(config["min_place_num"], config["max_place_num"] + 1)

        min_trans_offset = -3 if place_num > 3 else 1 - place_num
        trans_num = place_num + np.random.randint(min_trans_offset, 1)
        trans_num = max(1, trans_num)

        petri_matrix = PeGen.generate_random_petri_net(place_num, trans_num)
        if config["prune_flag"]:
            petri_matrix = PeGen.prune_petri_net(petri_matrix)
        if config["add_token"]:
            petri_matrix = PeGen.add_token_to_random_place(petri_matrix)

        results_dict, success = SPN.filter_stochastic_petri_net(
            petri_matrix,
            config["place_upper_bound"],
            config["marks_lower_limit"],
            config["marks_upper_limit"],
        )
        if success:
            return results_dict
    return None


def augment_single_sample(
    sample: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Applies data augmentation to a single SPN sample.

    Args:
        sample: The original SPN sample dictionary.
        config: A dictionary containing configuration parameters.

    Returns:
        A list of augmented SPN sample dictionaries.
    """
    if not sample or "petri_net" not in sample:
        return []

    augmented_samples = DataTransformation.transformation(
        sample["petri_net"],
        config["place_upper_bound"],
        config["marks_lower_limit"],
        config["marks_upper_limit"],
    )

    if not augmented_samples:
        return []

    # If more samples are generated than required, sample a subset.
    max_transforms = config["maxtransform_num"]
    if len(augmented_samples) > max_transforms:
        indices = np.random.choice(len(augmented_samples), max_transforms, replace=False)
        return [augmented_samples[i] for i in indices]

    return augmented_samples


def write_to_hdf5(
    h5_file: h5py.File, samples: List[Dict[str, Any]], start_idx: int
) -> int:
    """
    Writes a list of samples to an HDF5 file.

    Args:
        h5_file: The opened HDF5 file object.
        samples: A list of SPN sample dictionaries to write.
        start_idx: The starting index for naming the sample groups.

    Returns:
        The next available sample index.
    """
    dataset_group = h5_file["dataset_samples"]
    for i, sample_dict in enumerate(tqdm(samples, desc="Writing Samples to HDF5")):
        sample_group = dataset_group.create_group(f"sample_{start_idx + i:07d}")
        for key, value in sample_dict.items():
            try:
                np_value = np.array(value)
                # Apply compression only to non-scalar datasets
                if np_value.ndim > 0:
                    sample_group.create_dataset(
                        key, data=np_value, compression="gzip", compression_opts=4
                    )
                else:
                    sample_group.create_dataset(key, data=np_value)
            except TypeError:
                # Fallback for data that cannot be converted to a NumPy array
                sample_group.create_dataset(key, data=str(value))
    return start_idx + len(samples)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Sets up the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Generate Stochastic Petri Net (SPN) datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/DataConfig/SPNGenerate.json",
        help="Path to the configuration JSON file.",
    )
    return parser


def main():
    """Main function to generate the SPN dataset."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    config = DU.load_json(args.config)
    print("Configuration loaded:", config)

    # Prepare output directory and file path
    output_dir = os.path.join(config["write_data_loc"], "data_hdf5")
    DU.mkdir(output_dir)
    hdf5_filepath = os.path.join(output_dir, "spn_dataset.hdf5")
    print(f"Output HDF5 file will be: {hdf5_filepath}")

    with h5py.File(hdf5_filepath, "w") as hf:
        hf.attrs["generation_config"] = json.dumps(config, indent=4)
        hf.create_group("dataset_samples")

        # Generate initial set of SPNs in parallel
        print(f"Generating {config['data_num']} initial SPN samples...")
        initial_results = Parallel(n_jobs=config["parallel_job"])(
            delayed(generate_single_spn)(config)
            for _ in tqdm(range(config["data_num"]), desc="Initial Sample Generation")
        )
        initial_samples = [res for res in initial_results if res is not None]
        print(f"Successfully generated {len(initial_samples)} initial valid samples.")

        if not config["transformation_flag"]:
            # Write initial samples directly if no augmentation is needed
            total_written = write_to_hdf5(hf, initial_samples, 0)
        else:
            # Augment the initial samples in parallel
            print(f"Augmenting {len(initial_samples)} initial samples...")
            augmented_lists = Parallel(n_jobs=config["parallel_job"])(
                delayed(augment_single_sample)(sample, config)
                for sample in tqdm(initial_samples, desc="Augmenting Samples")
            )
            # Flatten the list of lists and write to HDF5
            all_augmented_samples = [item for sublist in augmented_lists for item in sublist]
            total_written = write_to_hdf5(hf, all_augmented_samples, 0)

        hf.attrs["total_samples_written"] = total_written
        print(f"Total samples written to HDF5: {total_written}")

    print(f"HDF5 file '{hdf5_filepath}' successfully written and closed.")


if __name__ == "__main__":
    main()
