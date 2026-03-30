#!/usr/bin/env python
"""
This script calculates and saves statistics about the distribution of data in a dataset.
"""

from pathlib import Path
import numpy as np
from spn_datasets.utils import DataUtil as DU
from spn_datasets.utils.ExcelTools import ExcelTool


def get_grid_boundaries(data_type):
    """Gets the grid boundaries based on the data type.

    Args:
        data_type (str): The type of data, either "GridData" or "RandData".

    Returns:
        tuple: A tuple containing the row and column boundaries.
    """
    if data_type == "RandData":
        row_boundaries = [5 + 2 * (i + 1) for i in range(3)]
        col_boundaries = [4 + 6 * (i + 1) for i in range(8)]
    else:
        row_boundaries = [5 + 2 * (i + 1) for i in range(5)]
        col_boundaries = [4 + 4 * (i + 1) for i in range(10)]
    return row_boundaries, col_boundaries


def count_data_distribution(data, row_boundaries, col_boundaries):
    """Counts the distribution of data across the grid.

    Args:
        data (dict): A dictionary of data instances.
        row_boundaries (list): The boundaries for the rows (places).
        col_boundaries (list): The boundaries for the columns (markings).

    Returns:
        np.ndarray: A 2D array representing the distribution count.
    """
    distribution_count = np.zeros((len(row_boundaries), len(col_boundaries)), dtype=int)
    for item in data.values():
        num_places = len(item["petri_net"])
        num_markings = len(item["arr_vlist"])
        row_idx = DU.get_lowest_index(num_places, row_boundaries)
        col_idx = DU.get_lowest_index(num_markings, col_boundaries)
        distribution_count[row_idx - 1, col_idx - 1] += 1
    return distribution_count


def save_statistics_to_excel(save_dir, dataset_name, row_boundaries, col_boundaries, distribution_count):
    """Saves the dataset statistics to an Excel file.

    Args:
        save_dir (str): The directory to save the Excel file in.
        dataset_name (str): The name of the dataset.
        row_boundaries (list): The boundaries for the rows.
        col_boundaries (list): The boundaries for the columns.
        distribution_count (np.ndarray): The distribution count matrix.
    """
    DU.create_directory(save_dir)
    excel_tool = ExcelTool(save_dir, f"static_{dataset_name}.xlsx", dataset_name)

    excel_tool.write_xls([["Data Distribution"]])
    excel_tool.write_xls_append([row_boundaries])
    excel_tool.write_xls_append([col_boundaries])
    excel_tool.write_xls_append(distribution_count.astype(int).tolist())

    excel_tool.read_excel_xls()


def main():
    """Main function to calculate and save dataset statistics."""
    dataset_name = "DS3"  # Example: "DS1" through "DS5"
    data_type = "GridData"  # "GridData" or "RandData"

    data_dir = Path(f"Data/{data_type}/{dataset_name}/ori_data")
    save_dir = Path(f"result/{data_type}/excel")

    train_data = DU.load_json_file(data_dir / "train_data.json")
    test_data = DU.load_json_file(data_dir / "test_data.json")

    row_boundaries, col_boundaries = get_grid_boundaries(data_type)

    train_distribution = count_data_distribution(train_data, row_boundaries, col_boundaries)
    test_distribution = count_data_distribution(test_data, row_boundaries, col_boundaries)
    total_distribution = train_distribution + test_distribution

    print("Data Distribution:")
    print(total_distribution)

    save_statistics_to_excel(save_dir, dataset_name, row_boundaries, col_boundaries, total_distribution)

    print(f"Total number of data points: {np.sum(total_distribution)}")


if __name__ == "__main__":
    main()
