"""
This script provides an interactive command-line interface to set up and run
a series of SPN (Stochastic Petri Net) generation tasks. It guides the user
through creating one or more scenarios, generating the necessary configuration
files for each, and then executing the generation scripts sequentially.
"""

import os
import subprocess
import toml
import shutil
from datetime import datetime
from tqdm import tqdm

import SPNGenerate


def load_default_configs():
    """Loads the default configurations from the project's config files."""
    spn_generate_defaults = toml.load("config/DataConfig/SPNGenerate.toml")
    partition_grid_defaults = toml.load("config/DataConfig/PartitionGrid.toml")
    return spn_generate_defaults, partition_grid_defaults


def get_generation_mode():
    """Asks the user to select the generation mode."""
    print("\n--- Select Generation Mode ---")
    while True:
        mode = input("Select generation mode (random/grid) [default: grid]: ").lower()
        if not mode:
            return "grid"
        if mode in ["random", "grid"]:
            return mode
        print("Invalid mode. Please choose 'random' or 'grid'.")


def get_common_data_folder():
    """Asks the user for a common data folder."""
    print("\n--- Common Data Folder ---")
    return get_user_input(
        "Enter the common data folder path",
        "data",
        str,
        "This folder will be used for all inputs, outputs, and temporary files.",
    )


def get_user_input(prompt, default, type_cast=str, help_text=""):
    """Gets user input with a a default value and optional help text."""
    if help_text:
        print(f"  {help_text}")

    # For boolean, we want a y/n prompt
    if type_cast == bool:
        user_input = input(f"{prompt} (y/n) [default: {'y' if default else 'n'}]: ")
        if not user_input:
            return default
        return user_input.lower() == "y"

    user_input = input(f"{prompt} [default: {default}]: ")
    if not user_input:
        return default

    while True:
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a value of type {type_cast.__name__}.")
            user_input = input(f"{prompt} [default: {default}]: ")
            if not user_input:
                return default


import numpy as np


def get_user_input_for_list(prompt, default, help_text=""):
    """
    Gets user input that should evaluate to a list of integers, allowing for
    Python expressions like lists, ranges, or list comprehensions.
    """
    if help_text:
        print(f"  {help_text}")

    while True:
        user_input_str = input(f"{prompt} [default: {default}]: ")
        if not user_input_str:
            user_input_str = default

        try:
            # Safely evaluate the input string.
            # The environment provides access to common built-ins like range() and list().
            result = eval(user_input_str, {"__builtins__": __builtins__})

            # If it's a generator or other iterable, consume it into a list
            if hasattr(result, "__iter__") and not isinstance(result, (list, str)):
                result = list(result)

            if not isinstance(result, list):
                print("Invalid input. Input must evaluate to a list, generator, or list comprehension.")
                continue

            # Check if all elements are integers
            if all(isinstance(x, int) for x in result):
                return result
            else:
                print("Invalid input. All elements in the list must be integers.")

        except Exception as e:
            print(f"Error evaluating input: {e}")
            print("Please enter a valid Python expression (e.g., [1, 2, 3] or range(5)).")


def _generate_grid_boundaries(min_val, max_val, num_bins):
    """Generates a list of boundaries for a grid."""
    if num_bins <= 0:
        return []
    # We use num_bins - 1 because if you have N bins, you have N-1 boundaries.
    return np.linspace(min_val, max_val, num_bins + 1, dtype=int)[1:-1].tolist()


def get_spn_generate_config(defaults, common_data_folder, generation_mode):
    """Interactively gets the configuration for SPNGenerate.py."""
    print("\n--- Configuring SPN Data Generation ---")
    config = {}
    config["output_data_location"] = os.path.join(common_data_folder, "raw")
    print(f"  Output data location is set to: {config['output_data_location']}")

    # In grid mode, the output must be a jsonl file for the next step.
    if generation_mode == "grid":
        config["output_file"] = "spn_dataset.jsonl"
        config["output_format"] = "jsonl"
        print("  Output file is set to 'spn_dataset.jsonl' for grid processing.")
    else:
        config["output_file"] = get_user_input("Output filename", "spn_dataset.jsonl", str, "Name of the output file.")
        config["output_format"] = get_user_input("Output format", "jsonl", str, "File format for the output data.")

    if generation_mode == "random":
        config["dataset_sizes"] = get_user_input_for_list(
            "List of dataset sizes (e.g., [1000, 5000] or range(1000, 5001, 1000))",
            "[1000, 5000]",
            "A Python list, generator, or list comprehension for dataset sizes.",
        )
    else:  # grid mode
        config["number_of_samples_to_generate"] = get_user_input(
            "Number of raw samples to generate",
            defaults["number_of_samples_to_generate"],
            int,
            "The number of raw SPN samples to generate before grid partitioning.",
        )

    config["number_of_parallel_jobs"] = get_user_input(
        "Number of parallel jobs",
        defaults["number_of_parallel_jobs"],
        int,
        "The number of parallel jobs to use for data generation.",
    )
    config["minimum_number_of_places"] = get_user_input(
        "Minimum number of places",
        defaults["minimum_number_of_places"],
        int,
        "The minimum number of places in the SPN.",
    )
    config["maximum_number_of_places"] = get_user_input(
        "Maximum number of places",
        defaults["maximum_number_of_places"],
        int,
        "The maximum number of places in the SPN.",
    )
    config["minimum_number_of_transitions"] = get_user_input(
        "Minimum number of transitions",
        defaults["minimum_number_of_transitions"],
        int,
        "The minimum number of transitions in the SPN.",
    )
    config["maximum_number_of_transitions"] = get_user_input(
        "Maximum number of transitions",
        defaults["maximum_number_of_transitions"],
        int,
        "The maximum number of transitions in the SPN.",
    )
    config["place_upper_bound"] = get_user_input(
        "Place upper bound", defaults["place_upper_bound"], int, "The upper bound for the number of places in the SPN."
    )
    config["marks_lower_limit"] = get_user_input(
        "Markings lower limit",
        defaults["marks_lower_limit"],
        int,
        "The lower limit for the number of markings in the SPN.",
    )
    config["marks_upper_limit"] = get_user_input(
        "Markings upper limit",
        defaults["marks_upper_limit"],
        int,
        "The upper limit for the number of markings in the SPN.",
    )
    config["enable_pruning"] = get_user_input(
        "Enable pruning", defaults["enable_pruning"], bool, "A boolean indicating whether to prune the SPN."
    )
    config["enable_token_addition"] = get_user_input(
        "Enable token addition",
        defaults["enable_token_addition"],
        bool,
        "A boolean indicating whether to add tokens to the SPN.",
    )
    config["enable_transformations"] = get_user_input(
        "Enable transformations",
        defaults["enable_transformations"],
        bool,
        "A boolean indicating whether to apply transformations to the SPNs.",
    )
    if config["enable_transformations"]:
        config["maximum_transformations_per_sample"] = get_user_input(
            "Maximum transformations per sample",
            defaults["maximum_transformations_per_sample"],
            int,
            "The maximum number of transformations to apply to each SPN.",
        )
    else:
        # Ensure this key exists to avoid errors, even if it's not used
        config["maximum_transformations_per_sample"] = defaults.get("maximum_transformations_per_sample", 1)

    config["enable_visualization"] = False
    config["visualization_output_location"] = defaults["visualization_output_location"]

    return config


def main():
    """
    Main function to run the interactive SPN generation setup.
    """
    print("Welcome to the interactive SPN generation setup.")

    common_data_folder = get_common_data_folder()
    temp_grid_folder = ""

    if os.path.exists("temp_configs"):
        shutil.rmtree("temp_configs")
    os.makedirs("temp_configs")

    if not os.path.exists("completed_configs"):
        os.makedirs("completed_configs")

    spn_defaults, grid_defaults = load_default_configs()

    scenario_count = 1
    generation_modes = {}  # To store generation mode for each scenario

    while True:
        add_scenario_input = input("\nAdd a new scenario? (y/n) [default: y]: ")
        if add_scenario_input.lower() == "n":
            if scenario_count == 1:
                print("No scenarios configured. Exiting.")
                return
            break

        generation_mode = get_generation_mode()

        if generation_mode == "random":
            num_created = configure_random_scenarios(spn_defaults, common_data_folder, generation_mode, scenario_count)
            # Store the mode for each created scenario
            for i in range(num_created):
                generation_modes[f"scenario_{scenario_count + i}"] = generation_mode
            scenario_count += num_created
        else:  # grid mode
            scenario_name = f"scenario_{scenario_count}"
            print(f"\n--- Configuring Scenario {scenario_count} ({generation_mode} mode) ---")
            temp_grid_folder = configure_grid_scenario(
                spn_defaults,
                grid_defaults,
                common_data_folder,
                generation_mode,
                scenario_name,
            )
            generation_modes[scenario_name] = generation_mode
            scenario_count += 1

    print("\nAll scenarios configured.")

    # --- Execution Phase ---
    print("\n--- Starting Execution Phase ---")
    scenarios_to_run = sorted(os.listdir("temp_configs"))
    for scenario_name in tqdm(scenarios_to_run, desc="Running Scenarios"):
        scenario_dir = os.path.join("temp_configs", scenario_name)
        if not os.path.isdir(scenario_dir):
            continue

        print(f"\n--- Running Scenario: {scenario_name} ---")
        run_scenario(scenario_dir, scenario_name, generation_modes[scenario_name])

    print("\nAll scenarios have been processed.")
    # Clean up temp directories
    shutil.rmtree("temp_configs")
    if temp_grid_folder and os.path.exists(temp_grid_folder):
        print(f"Cleaning up temporary grid folder: {temp_grid_folder}")
        shutil.rmtree(temp_grid_folder)


def configure_random_scenarios(spn_defaults, common_data_folder, generation_mode, scenario_count_offset):
    """Configures multiple scenarios for random generation based on a list of dataset sizes."""
    base_spn_config = get_spn_generate_config(spn_defaults, common_data_folder, generation_mode)
    dataset_sizes = base_spn_config.pop("dataset_sizes")

    print(f"\n--- Creating {len(dataset_sizes)} scenarios based on dataset sizes ---")
    for i, size in enumerate(dataset_sizes):
        scenario_name = f"scenario_{scenario_count_offset + i}"
        scenario_dir = os.path.join("temp_configs", scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        spn_config = base_spn_config.copy()
        spn_config["number_of_samples_to_generate"] = size
        # Create a more descriptive output filename for each scenario
        original_filename = spn_config.get("output_file", "spn_dataset.jsonl")
        name, ext = os.path.splitext(original_filename)
        spn_config["output_file"] = f"{name}_{size}_samples{ext}"

        with open(os.path.join(scenario_dir, "SPNGenerate.toml"), "w") as f:
            toml.dump(spn_config, f)

        print(f"Configuration for scenario {scenario_name} (size: {size}) saved in {scenario_dir}")
    return len(dataset_sizes)


def configure_grid_scenario(spn_defaults, grid_defaults, common_data_folder, generation_mode, scenario_name):
    """Interactively configures a scenario for grid-based generation with a simplified workflow."""
    scenario_dir = os.path.join("temp_configs", scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # --- Part 1: Get SPN Generation Config ---
    spn_config = get_spn_generate_config(spn_defaults, common_data_folder, generation_mode)
    with open(os.path.join(scenario_dir, "SPNGenerate.toml"), "w") as f:
        toml.dump(spn_config, f)

    # --- Part 2: Get Simplified Grid Config & Derive Parameters ---
    print("\n--- Configuring Grid Partitioning ---")
    grid_config = {}

    # Get simplified user inputs
    num_place_bins = get_user_input(
        "Number of grid bins for places", 5, int, "The number of divisions for the places axis of the grid."
    )
    num_marking_bins = get_user_input(
        "Number of grid bins for markings", 10, int, "The number of divisions for the markings axis of the grid."
    )
    grid_config["samples_per_grid"] = get_user_input(
        "Samples per grid cell",
        grid_defaults["samples_per_grid"],
        int,
        "The number of samples to take from each grid cell.",
    )
    grid_config["lambda_variations_per_sample"] = get_user_input(
        "Lambda variations per sample",
        grid_defaults["lambda_variations_per_sample"],
        int,
        "The number of lambda variations to generate for each sample.",
    )
    grid_config["accumulation_data"] = get_user_input(
        "Accumulate data in temporary grid",
        grid_defaults["accumulation_data"],
        bool,
        "If true, new raw data will be added to existing data in the temp folder.",
    )
    grid_config["output_format"] = get_user_input(
        "Final output format (hdf5 or jsonl)",
        grid_defaults["output_format"],
        str,
        "The format for the final grid dataset.",
    )
    grid_config["output_file"] = get_user_input(
        "Final output filename",
        grid_defaults["output_file"],
        str,
        "Name of the final grid dataset file.",
    )

    # Derive paths and boundaries
    spn_output_dir = os.path.join(spn_config["output_data_location"], f"data_{spn_config['output_format']}")
    grid_config["raw_data_location"] = os.path.join(spn_output_dir, spn_config["output_file"])
    grid_config["temporary_grid_location"] = os.path.join(common_data_folder, "temp_grid")
    grid_config["output_grid_location"] = os.path.join(common_data_folder, "grid")

    grid_config["places_grid_boundaries"] = _generate_grid_boundaries(
        spn_config["minimum_number_of_places"],
        spn_config["maximum_number_of_places"],
        num_place_bins,
    )
    grid_config["markings_grid_boundaries"] = _generate_grid_boundaries(
        spn_config["marks_lower_limit"],
        spn_config["marks_upper_limit"],
        num_marking_bins,
    )

    print(f"  Derived place boundaries: {grid_config['places_grid_boundaries']}")
    print(f"  Derived marking boundaries: {grid_config['markings_grid_boundaries']}")

    # Save the derived grid config
    with open(os.path.join(scenario_dir, "PartitionGrid.toml"), "w") as f:
        toml.dump(grid_config, f)

    print(f"\nConfiguration for {scenario_name} saved in {scenario_dir}")
    return grid_config.get("temporary_grid_location")


def run_scenario(scenario_dir, scenario_name, generation_mode):
    """Runs a single scenario, showing live output from the process."""
    spn_config_path = os.path.join(scenario_dir, "SPNGenerate.toml")

    # Run SPNGenerate.py
    print(f"Running SPNGenerate.py for {scenario_name}...")
    try:
        spn_config = toml.load(spn_config_path)
        SPNGenerate.run_generation_from_config(spn_config)
        print("SPNGenerate.py completed successfully.")
    except Exception as e:
        print(f"An error occurred while running SPNGenerate.py for {scenario_name}: {e}")
        return

    # Move SPN config file to completed_configs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_spn_config_name = f"{scenario_name}-{timestamp}-SPNGenerate.toml"
    shutil.move(spn_config_path, os.path.join("completed_configs", new_spn_config_name))

    if generation_mode == "grid":
        grid_config_path = os.path.join(scenario_dir, "PartitionGrid.toml")
        # Run ObtainGridDS.py
        print(f"Running ObtainGridDS.py for {scenario_name}...")
        grid_process = subprocess.run(
            ["python", "ObtainGridDS.py", "--config", grid_config_path], capture_output=True, text=True
        )
        if grid_process.returncode != 0:
            print(f"Error running ObtainGridDS.py for {scenario_name}.")
            print(grid_process.stderr)
            return
        print("ObtainGridDS.py completed successfully.")

        # Move grid config file to completed_configs
        new_grid_config_name = f"{scenario_name}-{timestamp}-PartitionGrid.toml"
        shutil.move(grid_config_path, os.path.join("completed_configs", new_grid_config_name))

    print(f"Configuration files for {scenario_name} moved to completed_configs.")
    print(f"--- Scenario {scenario_name} completed successfully! ---")


if __name__ == "__main__":
    main()
