# Use an official Python runtime as a parent image, as specified in the project's README.
FROM python:3.11-slim

# Set the working directory in the container to organize the project files.
WORKDIR /app

# Install uv, the recommended dependency management tool for this project.
# Using --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy the pyproject.toml file first to leverage Docker's build cache.
# This ensures that dependencies are only re-installed when pyproject.toml changes.
COPY pyproject.toml ./

# Install the project dependencies using uv sync.
# This command reads the pyproject.toml file and installs all required packages.
# --system-site-packages ensures that the packages are installed in the global environment.
RUN uv sync --system-site-packages

# Copy the rest of the application's code into the container.
COPY . .

# Set the default command to run the Cloud Run Job entrypoint.
# This expects SPN_CONFIG_JSON and GCS_BUCKET_NAME env vars to be set.
CMD ["python", "scripts/cloud_run_entrypoint.py"]
