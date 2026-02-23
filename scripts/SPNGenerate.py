#!/usr/bin/env python
"""
This script generates Stochastic Petri Net (SPN) datasets and saves them to HDF5.
"""

import argparse
import json
from pathlib import Path
import subprocess
import h5py
from tqdm import tqdm

from spn_datasets.generator.dataset_generator import DatasetGenerator
from spn_datasets.utils import DataUtil as DU
from spn_datasets.utils import FileWriter as FW


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
    config_path = Path(args.config)
    config = DU.load_toml_file(config_path) if config_path.exists() else {}
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def run_generation_from_config(config):
    """Runs the SPN generation process from a configuration dictionary."""
    output_format = config.get("output_format", "hdf5")
    output_dir = Path(config["output_data_location"]) / f"data_{output_format}"
    DU.create_directory(output_dir)
    output_path = output_dir / config["output_file"]

    generator = DatasetGenerator(config)
    all_samples = generator.generate_dataset()

    if not all_samples:
        print("No samples were generated. Skipping file writing and reporting.")
        return

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
        report_output_path = output_path.with_suffix(".html")
        try:
            result = subprocess.run(
                [
                    "python",
                    "generate_statistics.py",
                    "--input",
                    str(output_path),
                    "--output",
                    str(report_output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error generating statistical report for {output_path}:")
            print(e.stderr)


def main():
    """Main function to generate the SPN dataset."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    config = load_config(args)
    run_generation_from_config(config)


if __name__ == "__main__":
    main()
