"""
This script generates visualizations for a given dataset.
It creates visual representations of the Petri nets and their corresponding
Stochastic Petri Nets (SPNs) for both the training and testing sets.
"""

from pathlib import Path
from DataVisualization import Visual
from utils import DataUtil as DU


def visualize_dataset(dataset_type, dataset_name, num_parallel_jobs=-1):
    """Generates and saves visualizations for a dataset.

    Args:
        dataset_type (str): The type of the dataset (e.g., "RandData", "GridData").
        dataset_name (str): The name of the dataset (e.g., "DS1").
        num_parallel_jobs (int, optional): The number of parallel jobs to use
            for visualization. Defaults to -1 (using all available cores).
    """
    data_dir = Path(f"Data/{dataset_type}/{dataset_name}/ori_data/")
    pictures_dir = Path(f"Pics/{dataset_type}/{dataset_name}/")

    train_pics_dir = pictures_dir / "train_pics"
    test_pics_dir = pictures_dir / "test_pics"

    DU.create_directory(pictures_dir)
    DU.create_directory(train_pics_dir)
    DU.create_directory(test_pics_dir)

    train_data_path = data_dir / "train_data.json"
    test_data_path = data_dir / "test_data.json"

    if not train_data_path.exists() or not test_data_path.exists():
        print(f"Data not found for {dataset_type}/{dataset_name}. Skipping visualization.")
        return

    train_data = DU.load_json_file(train_data_path)
    test_data = DU.load_json_file(test_data_path)

    print(f"Visualizing training data for {dataset_name}...")
    Visual.visualize_dataset(train_data, train_pics_dir, num_parallel_jobs)

    print(f"Visualizing testing data for {dataset_name}...")
    Visual.visualize_dataset(test_data, test_pics_dir, num_parallel_jobs)

    print("Visualization complete.")


def main():
    """Main function to run the visualization script."""
    dataset_type = "RandData"
    dataset_name = "DS1"
    num_parallel_jobs = -1  # Use all available CPU cores

    visualize_dataset(dataset_type, dataset_name, num_parallel_jobs)


if __name__ == "__main__":
    main()
