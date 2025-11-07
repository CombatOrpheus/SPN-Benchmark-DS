"""
Tests for the GNN hyperparameter optimization script.
"""

import os
import toml
import runpy
from unittest.mock import patch, MagicMock

def test_optimize_gnn_script_runs(tmp_path):
    """
    Test that the optimize_gnn.py script runs without errors.
    """
    config_path = tmp_path / "config.toml"
    output_path = tmp_path / "best_params.toml"
    hdf5_path = "tests/temp_data/data_hdf5/test_data.h5"  # Using a small test dataset

    # Create a dummy config file
    config_data = {
        "training": {
            "epochs": 1,
        },
        "dataset": {
            "hdf5_path": hdf5_path,
            "graph_type": "reachability",
            "train_val_test_split": [0.8, 0.1, 0.1],
        },
    }
    with open(config_path, "w") as f:
        toml.dump(config_data, f)

    # Mock the study object that is created and used in the script
    with patch("optuna.create_study") as mock_create_study:
        # Create a mock study object
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study

        # Set up the best_trial attribute on the mock study
        mock_trial = MagicMock()
        mock_trial.value = 0.1
        mock_trial.params = {"model_name": "GCN", "learning_rate": 0.001, "batch_size": 32}
        mock_study.best_trial = mock_trial

        # Run the script with command-line arguments using runpy
        script_path = "scripts/optimize_gnn.py"
        args = [
            script_path,
            "--config", str(config_path),
            "--n-trials", "1",
            "--output-file", str(output_path)
        ]
        with patch("sys.argv", args):
            runpy.run_path(script_path, run_name="__main__")

    # Check that the output file was created
    assert os.path.exists(output_path)

    # Check the content of the output file
    best_params = toml.load(output_path)
    assert "training" in best_params
    assert best_params["training"] == {"model_name": "GCN", "learning_rate": 0.001, "batch_size": 32}
