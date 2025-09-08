"""
This script generates Stochastic Petri Net (SPN) datasets and saves them to HDF5.
"""

import argparse
import json
import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from DataGenerate import DataTransformation as DT
from DataGenerate import PetriGenerate as PeGen
from DataGenerate import SPN
from utils import DataUtil as DU


def generate_single_spn(config):
    """Generates a single SPN sample.

    Args:
        config (dict): Configuration dictionary for SPN generation.

    Returns:
        dict or None: A dictionary containing the SPN data, or None if generation fails.
    """
    max_attempts = 100
    for _ in range(max_attempts):
        place_num = np.random.randint(config["min_place_num"], config["max_place_num"] + 1)
        trans_offset = -3 if place_num > 3 else 1 - place_num
        trans_num = place_num + np.random.randint(trans_offset, 1)
        trans_num = max(1, trans_num)

        petri_matrix = PeGen.generate_random_petri_net(place_num, trans_num)
        if config.get("prune_flag"):
            petri_matrix = PeGen.prune_petri_net(petri_matrix)
        if config.get("add_token"):
            petri_matrix = PeGen.add_tokens_randomly(petri_matrix)

        results, success = SPN.filter_spn(
            petri_matrix,
            config["place_upper_bound"],
            config["marks_lower_limit"],
            config["marks_upper_limit"],
        )
        if success:
            return results
    return None


def augment_single_spn(sample, config):
    """Wrapper for data transformation to be used with joblib.

    Args:
        sample (dict): The original SPN sample.
        config (dict): Configuration for augmentation.

    Returns:
        list: A list of augmented samples.
    """
    if not sample or "petri_net" not in sample:
        return []

    petri_net = np.array(sample["petri_net"], dtype="long")
    augmented_data = DT.generate_petri_net_variations(
        petri_net,
        config["place_upper_bound"],
        config["marks_lower_limit"],
        config["marks_upper_limit"],
        config["parallel_job"],
    )

    if not augmented_data:
        return []

    max_transforms = config.get("maxtransform_num", len(augmented_data))
    if len(augmented_data) > max_transforms:
        indices = np.random.choice(len(augmented_data), max_transforms, replace=False)
        return [augmented_data[i] for i in indices]
    return augmented_data


def write_to_hdf5(group, data, compression="gzip", compression_opts=4):
    """Writes a sample to an HDF5 group."""
    for key, value in data.items():
        try:
            np_value = np.array(value)
            d_shape = np_value.shape
            d_type = np_value.dtype

            if np_value.ndim > 0:
                group.create_dataset(
                    key,
                    data=np_value,
                    shape=d_shape,
                    dtype=d_type,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                group.create_dataset(key, data=np_value, shape=d_shape, dtype=d_type)

        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save key '{key}' for sample {group.name}. Error: {e}")


def setup_arg_parser():
    """Sets up the argument parser with organized groups."""
    parser = argparse.ArgumentParser(
        description="Generate Stochastic Petri Net (SPN) datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config", type=str, default="config/DataConfig/SPNGenerate.json", help="Path to config JSON."
    )
    general_group.add_argument("--write_data_loc", type=str, help="Save directory.")
    general_group.add_argument("--output_file", type=str, help="Output HDF5 filename.")
    general_group.add_argument("--data_num", type=int, help="Number of samples.")
    general_group.add_argument("--parallel_job", type=int, help="Number of parallel jobs.")

    # SPN structure arguments
    spn_structure_group = parser.add_argument_group("SPN Structure")
    spn_structure_group.add_argument("--min_place_num", type=int, help="Min number of places.")
    spn_structure_group.add_argument("--max_place_num", type=int, help="Max number of places.")
    spn_structure_group.add_argument("--place_upper_bound", type=int, help="Upper bound for places.")
    spn_structure_group.add_argument("--marks_lower_limit", type=int, help="Lower limit for markings.")
    spn_structure_group.add_argument("--marks_upper_limit", type=int, help="Upper limit for markings.")

    # Generation process arguments
    generation_process_group = parser.add_argument_group("Generation Process")
    generation_process_group.add_argument("--prune_flag", action="store_true", help="Enable pruning.")
    generation_process_group.add_argument("--add_token", action="store_true", help="Enable adding tokens.")

    # Transformation arguments
    transformation_group = parser.add_argument_group("Transformation")
    transformation_group.add_argument("--transformation_flag", action="store_true", help="Enable augmentation.")
    transformation_group.add_argument("--maxtransform_num", type=int, help="Max number of transformations.")

    return parser


def load_config(args):
    """Loads configuration from JSON and overrides with command-line arguments."""
    config = DU.load_json_file(args.config) if os.path.exists(args.config) else {}
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def main():
    """Main function to generate the SPN dataset."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    config = load_config(args)

    output_dir = os.path.join(config["write_data_loc"], "data_hdf5")
    DU.create_directory(output_dir)
    hdf5_path = os.path.join(output_dir, config.get("output_file", "spn_dataset.hdf5"))

    print(f"Generating {config['data_num']} initial SPN samples...")
    initial_samples = Parallel(n_jobs=config["parallel_job"], backend="loky")(
        delayed(generate_single_spn)(config) for _ in trange(config["data_num"])
    )
    valid_samples = [s for s in initial_samples if s is not None]
    print(f"Generated {len(valid_samples)} valid initial samples.")

    all_samples = []
    if config.get("transformation_flag"):
        print("Augmenting samples...")
        augmented_lists = Parallel(n_jobs=config["parallel_job"], backend="loky")(
            delayed(augment_single_spn)(sample, config) for sample in tqdm(valid_samples, desc="Augmenting")
        )
        for sample_list in augmented_lists:
            all_samples.extend(sample_list)
    else:
        all_samples = valid_samples

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["generation_config"] = json.dumps(config)
        dataset_group = hf.create_group("dataset_samples")

        print(f"Writing {len(all_samples)} samples to HDF5...")
        for i, sample in enumerate(tqdm(all_samples, desc="Writing to HDF5")):
            sample_group = dataset_group.create_group(f"sample_{i:07d}")
            write_to_hdf5(sample_group, sample)

        hf.attrs["total_samples_written"] = len(all_samples)

    print(f"HDF5 file '{hdf5_path}' created successfully.")


if __name__ == "__main__":
    main()
