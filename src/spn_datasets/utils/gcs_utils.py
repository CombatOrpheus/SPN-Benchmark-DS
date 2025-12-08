import os
from pathlib import Path
from google.cloud import storage

def upload_file_to_gcs(local_file_path, bucket_name, destination_blob_name=None):
    """
    Uploads a file to Google Cloud Storage.

    Args:
        local_file_path (str or Path): The path to the local file to upload.
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str, optional): The name of the blob in GCS.
                                               If None, the filename of local_file_path is used.

    Returns:
        str: The public URL of the uploaded blob (if public) or gs:// URI.
    """
    local_file_path = Path(local_file_path)
    if not local_file_path.exists():
        raise FileNotFoundError(f"File {local_file_path} not found.")

    if destination_blob_name is None:
        destination_blob_name = local_file_path.name

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Uploading {local_file_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(str(local_file_path))

    print(f"File uploaded to gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"
