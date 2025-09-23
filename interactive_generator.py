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


def get_spn_generate_config(defaults, common_data_folder, generation_mode):
    """Interactively gets the configuration for SPNGenerate.py."""
    print("\n--- Configuring SPNGenerate.py ---")
    config = {}
    config["output_data_location"] = os.path.join(common_data_folder, "raw")
    print(f"  Output data location is set to: {config['output_data_location']}")

    config["output_file"] = get_user_input(
        "Output filename",
        "spn_dataset.jsonl",
        str,
        "Name of the output file. Must be .jsonl to be used by the next script.",
    )
    config["output_format"] = "jsonl"  # Hardcoded to jsonl as ObtainGridDS.py expects json

    print("  Output format is set to 'jsonl' to be compatible with the next script.")

    if generation_mode == "random":
        dataset_sizes_str = get_user_input(
            "List of dataset sizes (comma-separated)",
            "1000, 5000",
            str,
            "A comma-separated list of dataset sizes to generate.",
        )
        config["dataset_sizes"] = [int(s.strip()) for s in dataset_sizes_str.split(",")]
    else:  # grid mode
        config["number_of_samples_to_generate"] = get_user_input(
            "Number of samples to generate",
            defaults["number_of_samples_to_generate"],
            int,
            "The number of SPN samples to generate for the grid.",
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
    config["maximum_transformations_per_sample"] = get_user_input(
        "Maximum transformations per sample",
        defaults["maximum_transformations_per_sample"],
        int,
        "The maximum number of transformations to apply to each SPN.",
    )
    config["enable_visualization"] = False
    config["visualization_output_location"] = defaults["visualization_output_location"]

    return config


def get_partition_grid_config(defaults, spn_config, common_data_folder):
    """Interactively gets the configuration for ObtainGridDS.py."""
    print("\n--- Configuring ObtainGridDS.py ---")
    config = {}

    # The raw_data_location is determined by the output of the SPNGenerate script
    spn_output_dir = os.path.join(spn_config["output_data_location"], f"data_{spn_config['output_format']}")
    config["raw_data_location"] = os.path.join(spn_output_dir, spn_config["output_file"])
    print(f"  Raw data location is set to: {config['raw_data_location']}")

    config["temporary_grid_location"] = os.path.join(common_data_folder, "temp_grid")
    print(f"  Temporary grid location is set to: {config['temporary_grid_location']}")

    config["output_grid_location"] = os.path.join(common_data_folder, "grid")
    print(f"  Output grid location is set to: {config['output_grid_location']}")

    config["accumulation_data"] = get_user_input(
        "Accumulate data",
        defaults["accumulation_data"],
        bool,
        "A boolean indicating whether to accumulate data or start fresh.",
    )

    places_boundaries_str = get_user_input(
        "Places grid boundaries (comma-separated)",
        "7, 9, 11, 13, 15",
        str,
        "Boundaries for grid rows (number of places). Example: 10,20,30",
    )
    config["places_grid_boundaries"] = [int(s.strip()) for s in places_boundaries_str.split(",")]

    markings_boundaries_str = get_user_input(
        "Markings grid boundaries (comma-separated)",
        "8, 12, 16, 20, 24, 28, 32, 36, 40, 44",
        str,
        "Boundaries for grid columns (number of markings). Example: 100,200,300",
    )
    config["markings_grid_boundaries"] = [int(s.strip()) for s in markings_boundaries_str.split(",")]

    config["samples_per_grid"] = get_user_input(
        "Samples per grid", defaults["samples_per_grid"], int, "The number of samples to take from each grid cell."
    )
    config["lambda_variations_per_sample"] = get_user_input(
        "Lambda variations per sample",
        defaults["lambda_variations_per_sample"],
        int,
        "The number of lambda variations to generate for each sample.",
    )
    config["output_format"] = get_user_input(
        "Output format (hdf5 or jsonl)",
        defaults["output_format"],
        str,
        "The format for the output data.",
    )
    config["output_file"] = get_user_input(
        "Output filename",
        defaults["output_file"],
        str,
        "Name of the output file.",
    )
    return config


def main():
    """
    Main function to run the interactive SPN generation setup.
    """
    print("Welcome to the interactive SPN generation setup.")

    generation_mode = get_generation_mode()
    common_data_folder = get_common_data_folder()
    temp_grid_folder = ""

    if os.path.exists("temp_configs"):
        shutil.rmtree("temp_configs")
    os.makedirs("temp_configs")

    if not os.path.exists("completed_configs"):
        os.makedirs("completed_configs")

    spn_defaults, grid_defaults = load_default_configs()

    if generation_mode == "random":
        configure_random_scenarios(spn_defaults, common_data_folder, generation_mode)
    else:  # grid mode
        temp_grid_folder = configure_grid_scenarios(spn_defaults, grid_defaults, common_data_folder, generation_mode)


def configure_random_scenarios(spn_defaults, common_data_folder, generation_mode):
    """Configures multiple scenarios for random generation based on a list of dataset sizes."""
    base_spn_config = get_spn_generate_config(spn_defaults, common_data_folder, generation_mode)
    dataset_sizes = base_spn_config.pop("dataset_sizes")

    print(f"\n--- Creating {len(dataset_sizes)} scenarios based on dataset sizes ---")
    for i, size in enumerate(dataset_sizes):
        scenario_name = f"scenario_{i + 1}"
        scenario_dir = os.path.join("temp_configs", scenario_name)
        os.makedirs(scenario_dir)

        spn_config = base_spn_config.copy()
        spn_config["number_of_samples_to_generate"] = size
        # Create a more descriptive output filename for each scenario
        original_filename = spn_config.get("output_file", "spn_dataset.jsonl")
        name, ext = os.path.splitext(original_filename)
        spn_config["output_file"] = f"{name}_{size}_samples{ext}"

        with open(os.path.join(scenario_dir, "SPNGenerate.toml"), "w") as f:
            toml.dump(spn_config, f)

        print(f"Configuration for scenario {scenario_name} (size: {size}) saved in {scenario_dir}")


def configure_grid_scenarios(spn_defaults, grid_defaults, common_data_folder, generation_mode):
    """Interactively configures one or more scenarios for grid-based generation."""
    temp_grid_folder = ""
    scenario_count = 1
    while True:
        add_scenario_input = input(f"\nAdd scenario {scenario_count}? (y/n) [default: y]: ")
        if add_scenario_input.lower() == "n":
            break

        print(f"\n--- Configuring Scenario {scenario_count} ---")

        scenario_dir = os.path.join("temp_configs", f"scenario_{scenario_count}")
        os.makedirs(scenario_dir)

        spn_config = get_spn_generate_config(spn_defaults, common_data_folder, generation_mode)
        with open(os.path.join(scenario_dir, "SPNGenerate.toml"), "w") as f:
            toml.dump(spn_config, f)

        grid_config = get_partition_grid_config(grid_defaults, spn_config, common_data_folder)
        temp_grid_folder = grid_config["temporary_grid_location"]  # Save for cleanup
        with open(os.path.join(scenario_dir, "PartitionGrid.toml"), "w") as f:
            toml.dump(grid_config, f)

        print(f"\nConfiguration for scenario {scenario_count} saved in {scenario_dir}")
        scenario_count += 1
    return temp_grid_folder

    print("\nAll scenarios configured.")

    # --- Execution Phase ---
    print("\n--- Starting Execution Phase ---")
    scenarios_to_run = sorted(os.listdir("temp_configs"))
    for scenario_name in tqdm(scenarios_to_run, desc="Running Scenarios"):
        scenario_dir = os.path.join("temp_configs", scenario_name)
        if not os.path.isdir(scenario_dir):
            continue

        print(f"\n--- Running Scenario: {scenario_name} ---")
        run_scenario(scenario_dir, scenario_name, generation_mode)

    print("\nAll scenarios have been processed.")
    # Clean up temp directories
    shutil.rmtree("temp_configs")
    if temp_grid_folder and os.path.exists(temp_grid_folder):
        print(f"Cleaning up temporary grid folder: {temp_grid_folder}")
        shutil.rmtree(temp_grid_folder)


def run_scenario(scenario_dir, scenario_name, generation_mode):
    """Runs a single scenario."""
    spn_config_path = os.path.join(scenario_dir, "SPNGenerate.toml")

    # Run SPNGenerate.py
    print(f"Running SPNGenerate.py for {scenario_name}...")
    spn_process = subprocess.run(
        ["python", "SPNGenerate.py", "--config", spn_config_path], capture_output=True, text=True
    )
    if spn_process.returncode != 0:
        print(f"Error running SPNGenerate.py for {scenario_name}.")
        print(spn_process.stderr)
        return
    print("SPNGenerate.py completed successfully.")

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
