import os
import json
import sys
import shutil
from pathlib import Path
from spn_datasets.generator.dataset_generator import run_generation_from_config
from spn_datasets.utils.gcs_utils import upload_file_to_gcs

def main():
    config_json = os.environ.get("SPN_CONFIG_JSON")
    bucket_name = os.environ.get("GCS_BUCKET_NAME")

    if not config_json:
        print("Error: SPN_CONFIG_JSON environment variable not set.")
        sys.exit(1)

    if not bucket_name:
        print("Error: GCS_BUCKET_NAME environment variable not set.")
        sys.exit(1)

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON config: {e}")
        sys.exit(1)

    # Force output location to a temporary directory
    temp_output_dir = Path("/tmp/spn_generation_output")
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir(parents=True)

    config["output_data_location"] = str(temp_output_dir)

    # Ensure parallel jobs setting respects container limits if not set?
    # joblib uses all cores by default if n_jobs=-1, but config usually specifies it.
    # We leave it to the config.

    print("Starting generation...")
    try:
        run_generation_from_config(config)
    except Exception as e:
        print(f"Generation failed: {e}")
        sys.exit(1)

    print("Generation complete. Uploading to GCS...")

    # Upload everything in the output directory
    # The generator creates a subdirectory like 'data_hdf5' or 'data_jsonl'
    # We want to upload everything under temp_output_dir

    for root, dirs, files in os.walk(temp_output_dir):
        for file in files:
            file_path = Path(root) / file
            # Create a relative path for the blob name to preserve structure
            relative_path = file_path.relative_to(temp_output_dir)
            destination_blob_name = str(relative_path)

            try:
                upload_file_to_gcs(file_path, bucket_name, destination_blob_name)
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")
                # Don't exit, try to upload other files

if __name__ == "__main__":
    main()
