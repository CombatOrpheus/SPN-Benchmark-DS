"""
This script generates Stochastic Petri Net (SPN) datasets and saves them to HDF5.
"""

import argparse
import json
import os
import subprocess
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from DataGenerate import DataTransformation as DT
from DataGenerate import PetriGenerate as PeGen
from DataGenerate import SPN
from utils import DataUtil as DU
from utils import FileWriter as FW


def generate_single_spn(config):
    """Generates a single SPN sample.

    Args:
        config (dict): Configuration dictionary for SPN generation.

    Returns:
        dict or None: A dictionary containing the SPN data, or None if generation fails.
    """
    max_attempts = 100
    for _ in range(max_attempts):
        place_num = np.random.randint(config["minimum_number_of_places"], config["maximum_number_of_places"] + 1)
        trans_num = np.random.randint(
            config["minimum_number_of_transitions"], config["maximum_number_of_transitions"] + 1
        )

        petri_matrix = PeGen.generate_random_petri_net(place_num, trans_num)
        if config.get("enable_pruning"):
            petri_matrix = PeGen.prune_petri_net(petri_matrix)
        if config.get("enable_token_addition"):
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
        config["number_of_parallel_jobs"],
    )

    if not augmented_data:
        return []

    max_transforms = config.get("maximum_transformations_per_sample", len(augmented_data))
    if len(augmented_data) > max_transforms:
        indices = np.random.choice(len(augmented_data), max_transforms, replace=False)
        return [augmented_data[i] for i in indices]
    return augmented_data


def setup_arg_parser():
    """Sets up the argument parser with organized groups."""
    parser = argparse.ArgumentParser(
        description="Generate Stochastic Petri Net (SPN) datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config", type=str, default="config/DataConfig/SPNGenerate.toml", help="Path to config TOML file."
    )
    general_group.add_argument("--output_data_location", type=str, help="Save directory.")
    general_group.add_argument("--output_file", type=str, help="Output filename.")
    general_group.add_argument("--output_format", type=str, choices=["hdf5", "jsonl"], help="Output file format.")
    general_group.add_argument("--number_of_samples_to_generate", type=int, help="Number of samples.")
    general_group.add_argument("--number_of_parallel_jobs", type=int, help="Number of parallel jobs.")

    # SPN structure arguments
    spn_structure_group = parser.add_argument_group("SPN Structure")
    spn_structure_group.add_argument("--minimum_number_of_places", type=int, help="Min number of places.")
    spn_structure_group.add_argument("--maximum_number_of_places", type=int, help="Max number of places.")
    spn_structure_group.add_argument("--minimum_number_of_transitions", type=int, help="Min number of transitions.")
    spn_structure_group.add_argument("--maximum_number_of_transitions", type=int, help="Max number of transitions.")
    spn_structure_group.add_argument("--place_upper_bound", type=int, help="Upper bound for places.")
    spn_structure_group.add_argument("--marks_lower_limit", type=int, help="Lower limit for markings.")
    spn_structure_group.add_argument("--marks_upper_limit", type=int, help="Upper limit for markings.")

    # Generation process arguments
    generation_process_group = parser.add_argument_group("Generation Process")
    generation_process_group.add_argument("--enable_pruning", action="store_true", help="Enable pruning.")
    generation_process_group.add_argument("--enable_token_addition", action="store_true", help="Enable adding tokens.")

    # Transformation arguments
    transformation_group = parser.add_argument_group("Transformation")
    transformation_group.add_argument("--enable_transformations", action="store_true", help="Enable augmentation.")
    transformation_group.add_argument(
        "--maximum_transformations_per_sample", type=int, help="Max number of transformations."
    )

    # Reporting arguments
    reporting_group = parser.add_argument_group("Reporting")
    reporting_group.add_argument(
        "--enable_statistics_report", action="store_true", help="Enable statistical report generation."
    )

    return parser


def load_config(args):
    """Loads configuration from TOML and overrides with command-line arguments."""
    config = DU.load_toml_file(args.config) if os.path.exists(args.config) else {}
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def main():
    """Main function to generate the SPN dataset."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    config = load_config(args)

    output_format = config.get("output_format", "hdf5")  # Default to hdf5
    output_dir = os.path.join(config["output_data_location"], f"data_{output_format}")
    DU.create_directory(output_dir)
    output_path = os.path.join(output_dir, config["output_file"])

    print(f"Generating {config['number_of_samples_to_generate']} initial SPN samples...")
    initial_samples = Parallel(n_jobs=config["number_of_parallel_jobs"], backend="loky")(
        delayed(generate_single_spn)(config) for _ in trange(config["number_of_samples_to_generate"])
    )
    valid_samples = [s for s in initial_samples if s is not None]
    print(f"Generated {len(valid_samples)} valid initial samples.")

    all_samples = []
    if config.get("enable_transformations"):
        print("Augmenting samples...")
        augmented_lists = Parallel(n_jobs=config["number_of_parallel_jobs"], backend="loky")(
            delayed(augment_single_spn)(sample, config) for sample in tqdm(valid_samples, desc="Augmenting")
        )
        for sample_list in augmented_lists:
            all_samples.extend(sample_list)
    else:
        all_samples = valid_samples

    if output_format == "hdf5":
        with h5py.File(output_path, "w") as hf:
            hf.attrs["generation_config"] = json.dumps(config, cls=FW.NumpyEncoder)
            dataset_group = hf.create_group("dataset_samples")

            print(f"Writing {len(all_samples)} samples to HDF5...")
            for i, sample in enumerate(tqdm(all_samples, desc="Writing to HDF5")):
                sample_group = dataset_group.create_group(f"sample_{i:07d}")
                FW.write_to_hdf5(sample_group, sample)
            hf.attrs["total_samples_written"] = len(all_samples)
        print(f"HDF5 file '{output_path}' created successfully.")

    elif output_format == "jsonl":
        with open(output_path, "w") as f:
            f.write(json.dumps(config, cls=FW.NumpyEncoder) + "\n")
            for sample in tqdm(all_samples, desc="Writing to JSONL"):
                FW.write_to_jsonl(f, sample)
        print(f"JSONL file '{output_path}' created successfully.")

    if config.get("enable_statistics_report"):
        print("Generating statistical report...")
        report_output_path = os.path.splitext(output_path)[0] + "_report.html"
        try:
            subprocess.run(
                ["python", "generate_statistics.py", "--input", output_path, "--output", report_output_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Statistical report saved to '{report_output_path}'")
        except subprocess.CalledProcessError as e:
            print(f"Error generating statistical report for {output_path}:")
            print(e.stderr)


if __name__ == "__main__":
    main()
