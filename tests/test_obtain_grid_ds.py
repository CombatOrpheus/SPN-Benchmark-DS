import pytest
import os
import shutil
import toml
import h5py
import json
import numpy as np

# Make sure the script can find the root modules
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import SPNGenerate
import ObtainGridDS
from utils import DataUtil as DU


class TestObtainGridDS:
    """Test suite for ObtainGridDS.py"""

    def setup_method(self):
        """Set up a temporary directory for test artifacts."""
        self.temp_dir = "tests/temp_test_data"
        self.raw_data_loc = os.path.join(self.temp_dir, "raw")
        self.grid_temp_loc = os.path.join(self.temp_dir, "temp_grid")
        self.output_loc = os.path.join(self.temp_dir, "grid")

        # Clean up previous runs
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Create directories
        os.makedirs(self.raw_data_loc, exist_ok=True)
        os.makedirs(self.grid_temp_loc, exist_ok=True)
        os.makedirs(self.output_loc, exist_ok=True)

        # Define default configurations for tests
        self.spn_config = {
            "output_data_location": self.raw_data_loc,
            "output_file": "raw_spn_data.jsonl",
            "output_format": "jsonl",
            "number_of_samples_to_generate": 10,
            "number_of_parallel_jobs": 1,
            "minimum_number_of_places": 5,
            "maximum_number_of_places": 10,
            "minimum_number_of_transitions": 3,
            "maximum_number_of_transitions": 8,
            "place_upper_bound": 10,
            "marks_lower_limit": 4,
            "marks_upper_limit": 20,
            "enable_pruning": True,
            "enable_token_addition": True,
            "enable_transformations": False,
        }

        self.grid_config = {
            "raw_data_location": os.path.join(self.raw_data_loc, "data_jsonl", self.spn_config["output_file"]),
            "temporary_grid_location": self.grid_temp_loc,
            "output_grid_location": self.output_loc,
            "accumulation_data": False,
            "places_grid_boundaries": [7, 9],
            "markings_grid_boundaries": [8, 12, 16],
            "samples_per_grid": 1,
            "lambda_variations_per_sample": 2,
            "output_format": "hdf5",
            "output_file": "grid_dataset.hdf5",
        }

    def teardown_method(self):
        """Tear down the temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_scaffolding_creation(self):
        """Tests that the setup and teardown methods work."""
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(self.raw_data_loc)
        assert os.path.exists(self.grid_temp_loc)
        assert os.path.exists(self.output_loc)

    def test_full_pipeline_execution(self):
        """
        Tests the full data generation and grid partitioning pipeline end-to-end.
        This test is expected to fail until the bug is fixed.
        """
        # --- Step 1: Generate Raw Data using SPNGenerate ---
        SPNGenerate.run_generation_from_config(self.spn_config)

        # Verify that the raw data file was created
        raw_data_output_path = self.grid_config["raw_data_location"]
        assert os.path.exists(raw_data_output_path), "Raw data file was not generated."

        # --- Step 2: Run ObtainGridDS with the generated data ---
        # ObtainGridDS.py is designed to be run as a script, so we'll call its main functions.
        ObtainGridDS.partition_data_into_grid(
            self.grid_config["temporary_grid_location"],
            self.grid_config["accumulation_data"],
            raw_data_output_path,
            self.grid_config,
        )

        processed_data = ObtainGridDS.sample_and_transform_data(self.grid_config)
        ObtainGridDS.package_dataset(self.grid_config, processed_data)

        # --- Step 3: Verify the final output ---
        final_output_path = os.path.join(self.output_loc, self.grid_config["output_file"])
        assert os.path.exists(final_output_path), "Final grid dataset was not created."

        # Check content of HDF5 file
        if self.grid_config["output_format"] == "hdf5":
            with h5py.File(final_output_path, "r") as f:
                assert "dataset_samples" in f
                assert f.attrs["total_samples_written"] > 0

    def test_get_grid_index(self):
        """Tests the get_grid_index function with various values."""
        boundaries = [10, 20, 30]
        # Test values within each bin
        assert ObtainGridDS.get_grid_index(5, boundaries) == 1
        assert ObtainGridDS.get_grid_index(10, boundaries) == 2
        assert ObtainGridDS.get_grid_index(15, boundaries) == 2
        assert ObtainGridDS.get_grid_index(20, boundaries) == 3
        assert ObtainGridDS.get_grid_index(25, boundaries) == 3
        # Test value on the upper boundary
        assert ObtainGridDS.get_grid_index(30, boundaries) == 4
        # Test value exceeding the upper boundary
        assert ObtainGridDS.get_grid_index(35, boundaries) == 4
        # Test with empty boundaries
        assert ObtainGridDS.get_grid_index(100, []) == 1

    def test_initialize_grid(self):
        """Tests that the grid directories are initialized correctly."""
        config = {"places_grid_boundaries": [10, 20], "markings_grid_boundaries": [15]}
        ObtainGridDS._initialize_grid(self.grid_temp_loc, False, config)

        # Expect 3 place bins (p1, p2, p3) and 2 marking bins (m1, m2)
        for i in range(1, 4):
            for j in range(1, 3):
                assert os.path.exists(os.path.join(self.grid_temp_loc, f"p{i}", f"m{j}"))

    def test_partition_data_into_grid_logic(self):
        """Tests the partitioning logic specifically."""
        # Create a dummy raw data file
        raw_data = {
            "sample1": {"petri_net": [0] * 5, "arr_vlist": [0] * 5},  # p=5, m=5  -> p1, m1
            "sample2": {"petri_net": [0] * 8, "arr_vlist": [0] * 10},  # p=8, m=10 -> p1, m1
            "sample3": {"petri_net": [0] * 12, "arr_vlist": [0] * 20},  # p=12, m=20-> p2, m2
            "sample4": {"petri_net": [0] * 25, "arr_vlist": [0] * 30},  # p=25, m=30-> p3, m2
        }
        raw_data_path = os.path.join(self.temp_dir, "dummy_raw.jsonl")

        # Create a mock JSONL file
        with open(raw_data_path, "w") as f:
            f.write(json.dumps({}) + "\n")  # header
            for sample in raw_data.values():
                f.write(json.dumps(sample) + "\n")

        grid_config = {
            "raw_data_location": raw_data_path,
            "temporary_grid_location": self.grid_temp_loc,
            "accumulation_data": False,
            "places_grid_boundaries": [10, 20],
            "markings_grid_boundaries": [15],
        }

        ObtainGridDS.partition_data_into_grid(
            grid_config["temporary_grid_location"],
            grid_config["accumulation_data"],
            grid_config["raw_data_location"],
            grid_config,
        )

        # Check that files were created in the correct cells
        assert len(os.listdir(os.path.join(self.grid_temp_loc, "p1", "m1"))) == 2
        assert len(os.listdir(os.path.join(self.grid_temp_loc, "p2", "m2"))) == 1
        assert len(os.listdir(os.path.join(self.grid_temp_loc, "p3", "m2"))) == 1
        # Check an empty cell
        assert len(os.listdir(os.path.join(self.grid_temp_loc, "p1", "m2"))) == 0

    def test_package_dataset_formats(self):
        """Tests that both HDF5 and JSONL packaging formats work."""
        dummy_data = [{"sample": i} for i in range(5)]

        # Test HDF5
        hdf5_config = self.grid_config.copy()
        hdf5_config["output_format"] = "hdf5"
        hdf5_config["output_file"] = "test.hdf5"
        ObtainGridDS.package_dataset(hdf5_config, dummy_data)
        hdf5_path = os.path.join(self.output_loc, "test.hdf5")
        assert os.path.exists(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            assert f.attrs["total_samples_written"] == 5

        # Test JSONL
        jsonl_config = self.grid_config.copy()
        jsonl_config["output_format"] = "jsonl"
        jsonl_config["output_file"] = "test.jsonl"
        ObtainGridDS.package_dataset(jsonl_config, dummy_data)
        jsonl_path = os.path.join(self.output_loc, "test.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
            # 1 header line + 5 data lines
            assert len(lines) == 6
