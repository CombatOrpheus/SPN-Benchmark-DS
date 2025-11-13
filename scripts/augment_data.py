#!/usr/bin/env python
"""
This script augments an existing SPN dataset based on a configuration file.
It applies specified transformations to each sample in the input dataset
and saves the newly generated variations to an output file.
"""

import argparse
from pathlib import Path
import json
import h5py
import numpy as np
import toml
from tqdm import tqdm
from joblib import Parallel, delayed

from spn_datasets.generator import data_transformation as DT
from spn_datasets.utils import data_util as DU
from spn_datasets.utils import file_writer as FW


def setup_arg_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Augment an SPN dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset file (HDF5 or JSONL).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the augmented dataset file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the augmentation TOML configuration file.",
    )
    return parser


def load_dataset(filepath):
    """Loads a full dataset from an HDF5 or JSONL file."""
    filepath = Path(filepath)
    file_ext = filepath.suffix
    samples = []
    print(f"Loading dataset from {filepath}...")
    if file_ext == ".hdf5":
        with h5py.File(filepath, "r") as hf:
            dataset_group = hf.get("dataset_samples")
            if dataset_group:
                for sample_name in tqdm(dataset_group, desc="Loading HDF5 samples"):
                    sample_data = {key: val[:] for key, val in dataset_group[sample_name].items()}
                    samples.append(sample_data)
    elif file_ext == ".jsonl":
        with open(filepath, "r") as f:
            next(f)  # Skip config line
            for line in tqdm(f, desc="Loading JSONL samples"):
                samples.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    print(f"Loaded {len(samples)} samples.")
    return samples


def save_dataset(filepath, samples, config):
    """Saves a dataset to an HDF5 or JSONL file."""
    filepath = Path(filepath)
    file_ext = filepath.suffix
    output_dir = filepath.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print(f"Saving {len(samples)} augmented samples to {filepath}...")
    if file_ext == ".hdf5":
        with h5py.File(filepath, "w") as hf:
            hf.attrs["augmentation_config"] = json.dumps(config, cls=FW.NumpyEncoder)
            dataset_group = hf.create_group("dataset_samples")
            for i, sample in enumerate(tqdm(samples, desc="Writing to HDF5")):
                sample_group = dataset_group.create_group(f"sample_{i:07d}")
                FW.write_to_hdf5(sample_group, sample)
            hf.attrs["total_samples_written"] = len(samples)
    elif file_ext == ".jsonl":
        with open(filepath, "w") as f:
            f.write(json.dumps(config, cls=FW.NumpyEncoder) + "\n")
            for sample in tqdm(samples, desc="Writing to JSONL"):
                FW.write_to_jsonl(f, sample)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    print("Save complete.")


def augment_sample(sample, config):
    """Wrapper function to apply augmentation to a single sample."""
    if "petri_net" not in sample:
        return []
    return DT.generate_petri_net_variations(sample["petri_net"], config)


def main():
    """Main function to augment the dataset."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    config = DU.load_toml_file(config_path)
    print(f"Augmentation config loaded from {config_path}")

    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return
    input_samples = load_dataset(input_path)

    if not input_samples:
        print("No samples found in the input dataset. Exiting.")
        return

    # Augment samples in parallel
    print("Starting data augmentation...")
    parallel_jobs = config.get("number_of_parallel_jobs", 1)
    augmented_lists = Parallel(n_jobs=parallel_jobs)(
        delayed(augment_sample)(sample, config) for sample in tqdm(input_samples, desc="Augmenting samples")
    )

    # Flatten the list of lists
    all_augmented_samples = [item for sublist in augmented_lists for item in sublist]
    print(f"Generated {len(all_augmented_samples)} new augmented samples.")

    # Save the new dataset
    save_dataset(args.output, all_augmented_samples, config)


if __name__ == "__main__":
    main()
