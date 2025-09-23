# Stochastic Petri Net (SPN) Dataset Generation

This project provides a powerful and flexible tool for generating benchmark datasets of Stochastic Petri Nets (SPNs). It is designed to facilitate research and development in areas such as SPN analysis, performance evaluation, and machine learning on graph-structured data.

The core of this project is the `SPNGenerate.py` script, which allows you to create large datasets of SPNs with varying structures and properties. The generated datasets are saved in the efficient HDF5 format, making them easy to use and manage.

## Features

- **Customizable SPN Generation**: Generate SPNs with configurable parameters, such as the number of places, transitions, and token configurations.
- **Efficient Data Storage**: Datasets are stored in the HDF5 format, which provides high performance and compression.
- **Easy-to-Use Data Reader**: A utility module, `utils.HDF5Reader`, is provided for easy access to the generated data.

## Getting Started

### Prerequisites

Before you begin, ensure you have Python 3.11 or higher installed. You will also need to install the required dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    This project uses `uv` for dependency management. You can install the required packages from `pyproject.toml`:
    ```bash
    pip install uv
    uv sync
    ```
    Alternatively, you can use `conda` with the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate spn-benchmark
    ```

### Using Docker

For a consistent and reproducible development environment, you can use the provided `Dockerfile`.

1.  **Build the Docker image:**
    From the root of the project, run the following command:
    ```bash
    docker build -t spn-benchmark .
    ```

2.  **Run the Docker container:**
    This command starts an interactive `bash` shell inside the container, with the project directory mounted at `/app`.
    ```bash
    docker run -it -v .:/app spn-benchmark
    ```
    You can now run any of the project's scripts from within the container's shell.

### Data Generation

The primary script for generating datasets is `SPNGenerate.py`. Its behavior is controlled by a TOML configuration file. A sample configuration is provided at `config/DataConfig/SPNGenerate.toml`.

To generate a dataset, run the following command:
```bash
python SPNGenerate.py --config config/DataConfig/SPNGenerate.toml
```
This will create an HDF5 file named `spn_dataset.hdf5` in the location specified in your configuration file.

### Reading the Generated Data

The `utils.HDF5Reader` module provides a convenient way to read the data from the generated HDF5 files. The `SPNDataReader` class allows you to iterate through the dataset and access individual samples.

Here's a simple example of how to use it:

```python
from utils.HDF5Reader import SPNDataReader

# Path to your HDF5 file
hdf5_path = "path/to/your/spn_dataset.hdf5"

# Use the reader as a context manager
with SPNDataReader(hdf5_path) as reader:
    # Get the total number of samples
    num_samples = len(reader)
    print(f"Total samples: {num_samples}")

    # Get a specific sample by index
    if num_samples > 0:
        sample = reader.get_sample(0)
        print("First sample keys:", list(sample.keys()))
        # Example: Accessing node features
        # node_features = sample['node_f']
        # print("Node features shape:", node_features.shape)

    # Iterate through all samples
    for i, sample in enumerate(reader):
        print(f"Processing sample {i}...")
        # Your processing logic here
        if i >= 4:  # Stop after 5 samples for this example
            break
```

## Project Structure

-   `DataGenerate/`: Contains the scripts for generating SPNs and their corresponding graphs.
-   `SPNGenerate.py`: The main script for generating datasets.
-   `config/DataConfig/`: Contains configuration files for data generation.
-   `utils/`: Provides utility modules, including the HDF5 data reader.
-   `pyproject.toml`: Defines project dependencies for `uv` and `pip`.
-   `environment.yml`: Defines the `conda` environment.
