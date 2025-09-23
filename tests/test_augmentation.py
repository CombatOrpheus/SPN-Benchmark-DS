import os
import subprocess
import tempfile
import sys
import h5py
import numpy as np
import toml
import json


def create_dummy_hdf5_dataset(filepath, num_samples=2):
    """Creates a small HDF5 dataset for testing augmentation."""
    config = {"original_data": True}
    with h5py.File(filepath, "w") as hf:
        hf.attrs["generation_config"] = json.dumps(config)
        dataset_group = hf.create_group("dataset_samples")
        for i in range(num_samples):
            sample_group = dataset_group.create_group(f"sample_{i:07d}")
            # Create a simple, valid-looking petri net
            petri_net = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=int)
            petri_net = np.hstack([petri_net, np.ones((petri_net.shape[0], 1), dtype=int)])
            sample_group.create_dataset("petri_net", data=petri_net)
            # Add other required fields for rate variations
            sample_group.create_dataset("arr_vlist", data=np.random.rand(5, 3))
            sample_group.create_dataset("arr_edge", data=np.random.rand(5, 2))
            sample_group.create_dataset("arr_tranidx", data=np.random.rand(5))


def create_dummy_augmentation_config(filepath):
    """Creates a TOML config file for testing augmentation."""
    config = {
        "number_of_parallel_jobs": 1,
        "enable_add_edge": True,
        "enable_delete_edge": True,
        "enable_add_place": True,
        "enable_add_token": True,
        "enable_delete_token": True,
        "enable_rate_variations": True,
        "num_rate_variations_per_structure": 2,
        "place_upper_bound": 10,
        "marks_lower_limit": 2,
        "marks_upper_limit": 100,
    }
    with open(filepath, "w") as f:
        toml.dump(config, f)


def test_augmentation_script_runs_successfully():
    """
    Tests that the augment_data.py script runs without errors
    and produces an augmented dataset.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "test_dataset.hdf5")
        output_file = os.path.join(tmpdir, "augmented_dataset.hdf5")
        config_file = os.path.join(tmpdir, "aug_config.toml")

        create_dummy_hdf5_dataset(input_file)
        create_dummy_augmentation_config(config_file)
        assert os.path.exists(input_file)
        assert os.path.exists(config_file)

        script_path = os.path.join(os.path.dirname(__file__), "..", "augment_data.py")

        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--input",
                input_file,
                "--output",
                output_file,
                "--config",
                config_file,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
        assert os.path.exists(output_file), "The augmented output file was not created."

        with h5py.File(output_file, "r") as hf:
            assert "dataset_samples" in hf, "Dataset group missing in output."
            # Check that some new samples were created. The exact number is non-deterministic.
            assert len(hf["dataset_samples"]) > 0, "No samples were generated in the output file."
            assert "augmentation_config" in hf.attrs, "Augmentation config missing from metadata."
