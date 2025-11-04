from pathlib import Path
import subprocess
import tempfile
import sys
import h5py
import numpy as np
import json


def create_dummy_hdf5_dataset(filepath, num_samples=5):
    """Creates a small HDF5 dataset for testing."""
    config = {
        "number_of_samples_to_generate": num_samples,
        "minimum_number_of_places": 5,
        "maximum_number_of_places": 10,
    }
    with h5py.File(filepath, "w") as hf:
        hf.attrs["generation_config"] = json.dumps(config)
        dataset_group = hf.create_group("dataset_samples")
        for i in range(num_samples):
            sample_group = dataset_group.create_group(f"sample_{i:07d}")
            places = np.random.randint(5, 11)
            transitions = np.random.randint(4, 9)
            states = np.random.randint(10, 51)
            petri_net = np.zeros((places, 2 * transitions + 1))
            arr_vlist = np.zeros((states, places))
            sample_group.create_dataset("petri_net", data=petri_net)
            sample_group.create_dataset("arr_vlist", data=arr_vlist)


def test_statistics_script_runs_successfully():
    """
    Tests that the generate_statistics.py script runs without errors
    and produces an HTML report.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_file = tmpdir / "test_dataset.hdf5"
        output_file = tmpdir / "report.html"

        create_dummy_hdf5_dataset(input_file)
        assert input_file.exists()

        script_path = Path(__file__).parent.parent / "generate_statistics.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--input", str(input_file), "--output", str(output_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
        assert output_file.exists(), "The HTML report file was not created."

        with open(output_file, "r") as f:
            content = f.read()
            assert len(content) > 0, "The HTML report is empty."
            assert "SPN Dataset Statistical Report" in content, "Report title is missing."
