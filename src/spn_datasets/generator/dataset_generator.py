import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from spn_datasets.generator import DataTransformation as DT
from spn_datasets.generator import PetriGenerate as PeGen
from spn_datasets.generator import SPN
from spn_datasets.utils import DataUtil as DU
from spn_datasets.utils import FileWriter as FW


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


def run_generation_from_config(config):
    """Runs the SPN generation process from a configuration dictionary."""
    output_format = config.get("output_format", "hdf5")
    output_dir = Path(config["output_data_location"]) / f"data_{output_format}"
    DU.create_directory(output_dir)
    output_path = output_dir / config["output_file"]

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
            # Note: This relies on generate_statistics.py being available in the same directory or path.
            # In a refactored setup, we might want to call a python function directly instead of subprocess.
            # For now, I'll keep subprocess but might need to adjust the path or command.
            result = subprocess.run(
                [
                    "python",
                    "scripts/generate_statistics.py", # Assumed relative to project root
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
