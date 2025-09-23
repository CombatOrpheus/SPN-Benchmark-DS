# Agent Instructions

This document provides instructions for AI agents working on this codebase.

## Project Overview

This project, "SPN-Benchmarks", is designed for generating and working with benchmark datasets for Stochastic Petri Nets (SPNs). The primary focus is on creating large, customizable datasets in an efficient format, suitable for research in SPN analysis and graph-based machine learning.

The key components are:
-   **Data Generation**: The `SPNGenerate.py` script is the main tool for creating datasets. It is configured via a TOML file (e.g., `config/DataConfig/SPNGenerate.toml`).
-   **Data Utilities**: The `utils/HDF5Reader.py` module provides a simple and efficient way to read the generated HDF5 datasets.
-   **Core Logic**: The `DataGenerate/` directory contains the underlying logic for creating SPN structures and converting them into graph representations.

While the project also contains modules for Graph Neural Networks (`GNNs/`), the primary focus of this document is on the data generation pipeline.

## Development Guidelines

### Dependencies

This project uses `pyproject.toml` as the single source of truth for Python dependencies.

-   **Primary Tool**: `uv` is the recommended tool for managing dependencies. To install, run:
    ```bash
    uv sync
    ```
-   **Conda**: For `conda` users, an `environment.yml` file is provided. To create the environment, run:
    ```bash
    conda env create -f environment.yml
    ```
    Keep this file in sync with `pyproject.toml` when adding new dependencies.
-   **Adding Dependencies**: New dependencies should be added to the `[project.dependencies]` section of `pyproject.toml`.

### Code Style

This project uses `black` for code formatting. Please format your code before submitting. The configuration is defined in `pyproject.toml`.

To format the entire project, run:
```bash
black .
```

### Testing

This project uses `pytest` for testing, run via `uv`. Before submitting changes, please ensure that all existing tests pass and that any new functionality is covered by new tests.

To run the tests, follow these steps:

1.  **Sync all dependencies, including development dependencies:**
    ```bash
    uv sync --all-extras
    ```

2.  **Install the project in editable mode:** This makes local modules like `DataGenerate` available to the test suite.
    ```bash
    uv pip install -e .
    ```

3.  **Run the test suite:**
    ```bash
    uv run pytest
    ```
