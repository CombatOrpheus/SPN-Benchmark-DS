import os
import json
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add scripts directory to path to import cloud_run_entrypoint
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from cloud_run_entrypoint import main

@patch("cloud_run_entrypoint.upload_file_to_gcs")
@patch("cloud_run_entrypoint.run_generation_from_config")
@patch.dict(os.environ, {
    "SPN_CONFIG_JSON": '{"test": "config"}',
    "GCS_BUCKET_NAME": "test-bucket"
})
def test_cloud_job_success(mock_run, mock_upload):
    # Mock run_generation to create a dummy file
    def side_effect(config):
        output_dir = config["output_data_location"]
        # Ensure dir exists (it should be created by main)
        os.makedirs(output_dir, exist_ok=True)

        # Write a dummy file
        p = os.path.join(output_dir, "test_file.h5")
        with open(p, "w") as f:
            f.write("dummy content")

    mock_run.side_effect = side_effect

    main()

    mock_run.assert_called_once()
    # Check if config was passed correctly
    config_arg = mock_run.call_args[0][0]
    assert config_arg["test"] == "config"
    assert "/tmp/spn_generation_output" in config_arg["output_data_location"]

    mock_upload.assert_called()
    # Check if arguments are correct
    # We might have multiple calls if directory traversal finds things, but we created one file.

    # Find the call that uploaded test_file.h5
    found = False
    for call in mock_upload.call_args_list:
        args = call[0]
        if "test_file.h5" in str(args[0]):
            assert args[1] == "test-bucket"
            assert args[2] == "test_file.h5"
            found = True
            break
    assert found
