"""This module provides utility functions for data handling, including file I/O,
data partitioning, and preprocessing.
"""

import json
from pathlib import Path
import toml
from sklearn.model_selection import train_test_split
import numpy as np
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Tuple,
)


def create_directory(path: Path) -> None:
    """Creates a directory if it does not already exist.

    Args:
        path: The path of the directory to create.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_data_from_txt(file_path: Path) -> np.ndarray:
    """Loads data from a comma-separated text file.

    Args:
        file_path: The path to the text file.

    Returns:
        The data loaded from the text file.
    """
    return np.loadtxt(file_path, delimiter=",")


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Loads data from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The data loaded from the JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Loads data from a JSONL file, skipping the header.

    Args:
        file_path: The path to the JSONL file.

    Yields:
        The data loaded from each line of the JSONL file.
    """
    with open(file_path, "r") as f:
        next(f)
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_toml_file(file_path: Path) -> Dict[str, Any]:
    """Loads data from a TOML file.

    Args:
        file_path: The path to the TOML file.

    Returns:
        The data loaded from the TOML file.
    """
    with open(file_path, "r") as f:
        return toml.load(f)


def load_all_data_from_json_directory(
    directory_path: Path,
) -> Generator[Dict[str, Any], None, None]:
    """Loads all JSON files from a directory and yields them one by one.

    Args:
        directory_path: The path to the directory containing the JSON files.

    Yields:
        The data loaded from each JSON file.
    """
    _, json_files = count_json_files(directory_path)
    for json_file in json_files:
        yield load_json_file(directory_path / json_file)


def count_json_files(directory_path: Path) -> Tuple[int, List[str]]:
    """Counts the number of JSON files in a directory.

    Args:
        directory_path: The path to the directory.

    Returns:
        A tuple containing the number of JSON files and a list of their names.
    """
    json_files = sorted(
        [p.name for p in directory_path.iterdir() if p.is_file() and p.suffix == ".json"],
        key=lambda x: int(x.replace("data", "").replace(".json", "")),
    )
    return len(json_files), json_files


def load_all_data_from_txt_directory(directory_path: Path) -> List[np.ndarray]:
    """Loads all text files from a directory into a list.

    Args:
        directory_path: The path to the directory containing the text files.

    Returns:
        A list of numpy.ndarray, each containing the data from a text file.
    """
    txt_files = get_all_txt_files_in_directory(directory_path)
    return [
        load_data_from_txt(directory_path / txt_file) for txt_file in txt_files
    ]


def write_data_to_txt(file_path: Path, data: np.ndarray) -> None:
    """Writes data to a comma-separated text file.

    Args:
        file_path: The path to the text file.
        data: The data to write.
    """
    np.savetxt(file_path, data, fmt="%.0f", delimiter=",")


def get_all_txt_files_in_subdirectories(directory_path: Path) -> List[Path]:
    """Gets a list of all text files in the subdirectories of a given directory.

    Args:
        directory_path: The path to the main directory.

    Returns:
        A list of full paths to all text files.
    """
    subdirectories = sorted(
        [d for d in directory_path.iterdir() if d.is_dir()],
        key=lambda x: int(x.name.replace("data", "")),
    )
    all_txt_files = []
    for subdir in subdirectories:
        txt_files = sorted(
            [f for f in subdir.iterdir() if f.is_file() and f.suffix == ".txt"],
            key=lambda x: int(x.stem.replace("data", "")),
        )
        full_paths = [subdir / f.name for f in txt_files]
        all_txt_files.extend(full_paths)
    return all_txt_files


def load_arrivable_graph_data(
    directory_path: Path, index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the data for an arrivable graph.

    Args:
        directory_path: The path to the directory containing the graph data.
        index: The index of the graph data to load.

    Returns:
        A tuple containing the node data, hu data, pb data, and transition number.
    """
    pb_data = load_data_from_txt(directory_path / f"pb_data{index}.txt").astype(int)
    node_data = load_data_from_txt(directory_path / f"node_data{index}.txt").astype(
        int
    )
    hu_data = load_data_from_txt(directory_path / f"hu_data{index}.txt").astype(int)
    tran_num = load_data_from_txt(directory_path / f"tran_data{index}.txt").astype(
        int
    )
    return node_data, hu_data, pb_data, tran_num


def save_data_to_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """Saves data to a JSON file.

    Args:
        file_path: The path to the JSON file.
        data: The data to save.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def get_all_txt_files_in_directory(directory_path: Path) -> List[str]:
    """Gets a list of all text files in a directory.

    Args:
        directory_path: The path to the directory.

    Returns:
        A list of text file names.
    """
    return sorted(
        [
            p.name
            for p in directory_path.iterdir()
            if p.is_file() and p.suffix == ".txt"
        ],
        key=lambda x: int(x.replace("data", "").replace(".txt", "")),
    )


def add_preprocessed_to_dict(
    node_feature_num: int,
    key: str,
    value: Dict[str, Any],
    preprocessed_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Adds preprocessed data to a dictionary.

    Args:
        node_feature_num: The number of node features.
        key: The key for the new dictionary entry.
        value: The original data dictionary.
        preprocessed_dict: The dictionary to add the preprocessed data to.

    Returns:
        The updated dictionary with the preprocessed data.
    """
    arrivable_dict = {}
    node_unit = []
    for row in value["arr_vlist"]:
        row_n = np.zeros(node_feature_num)
        row_n[: len(row)] = row
        node_unit.append(row_n.tolist())

    arrivable_dict["node_f"] = node_unit
    arrivable_dict["edge_index"] = value["arr_edge"]

    spn_lambda = np.array(value["spn_labda"])
    arr_tran_idx = [int(pa) for pa in value["arr_tranidx"]]
    arrivable_dict["edge_f"] = spn_lambda[arr_tran_idx].tolist()

    arrivable_dict["label"] = value["spn_mu"]
    preprocessed_dict[key] = arrivable_dict
    return preprocessed_dict


def sample_json_files_from_directory(
    num_samples: int, directory_path: Path
) -> List[Dict[str, Any]]:
    """Samples a specified number of JSON files from a directory.

    Args:
        num_samples: The number of JSON files to sample.
        directory_path: The path to the directory.

    Returns:
        A list of dictionaries, each loaded from a sampled JSON file.
    """
    if not directory_path.exists():
        return []

    _, json_files = count_json_files(directory_path)

    if not json_files:
        return []

    np.random.shuffle(json_files)

    sampled_data = []
    for file_name in json_files:
        if len(sampled_data) >= num_samples:
            break

        file_path = directory_path / file_name
        data = load_json_file(file_path)

        if -100 <= data["spn_mu"] <= 100:
            sampled_data.append(data)

    return sampled_data


def create_data_dictionary(all_data: List[Any]) -> Dict[str, Any]:
    """Creates a dictionary from a list of data.

    Args:
        all_data: A list of data items.

    Returns:
        A dictionary where keys are 'data1', 'data2', etc.
    """
    return {f"data{i+1}": data for i, data in enumerate(all_data)}


def load_arrivable_data_range(
    location_template: str, lower_limit: int, upper_limit: int
) -> List[np.ndarray]:
    """Loads a range of data files based on a template.

    Args:
        location_template: A string template for the file path.
        lower_limit: The starting index.
        upper_limit: The ending index.

    Returns:
        A list of loaded data.
    """
    return [
        load_data_from_txt(Path(location_template % str(i + 1)))
        for i in range(lower_limit, upper_limit)
    ]


def load_lambda_mu_range(
    location_template: str, i: int, lower_limit: int, upper_limit: int
) -> List[np.ndarray]:
    """Loads a range of lambda and mu data files.

    Args:
        location_template: A string template for the file path.
        i: The first index for the template.
        lower_limit: The starting second index.
        upper_limit: The ending second index.

    Returns:
        A list of loaded data.
    """
    return [
        load_data_from_txt(Path(location_template % (str(i), str(j + 1))))
        for j in range(lower_limit, upper_limit)
    ]


def _preprocess_and_save_data(
    data_dict: Dict[str, Any], node_feature_num: int, file_path: Path
) -> None:
    """Preprocesses data and saves it to a JSON file."""
    preprocessed_dict = {}
    for key, value in data_dict.items():
        preprocessed_dict = add_preprocessed_to_dict(
            node_feature_num, key, value, preprocessed_dict
        )
    save_data_to_json_file(file_path, preprocessed_dict)


def partition_datasets(
    json_data_directory: Path, node_feature_num: int, test_ratio: float = 0.2
) -> None:
    """Partitions the dataset into training and testing sets.

    Args:
        json_data_directory: The path to the directory with JSON data.
        node_feature_num: The number of node features.
        test_ratio: The ratio of data to be used for testing. Defaults to 0.2.
    """
    original_data_dir = json_data_directory / "ori_data"
    preprocessed_data_dir = json_data_directory / "preprocessd_data"

    all_data_path = original_data_dir / "all_data.json"
    all_data = load_json_file(all_data_path)

    train_data, test_data = train_test_split(
        list(all_data.values()), test_size=test_ratio, random_state=0
    )

    train_data_dict = create_data_dictionary(train_data)
    test_data_dict = create_data_dictionary(test_data)

    save_data_to_json_file(original_data_dir / "train_data.json", train_data_dict)
    save_data_to_json_file(original_data_dir / "test_data.json", test_data_dict)

    create_directory(preprocessed_data_dir)

    train_preprocessed_path = preprocessed_data_dir / "train_data.json"
    _preprocess_and_save_data(
        train_data_dict, node_feature_num, train_preprocessed_path
    )

    test_preprocessed_path = preprocessed_data_dir / "test_data.json"
    _preprocess_and_save_data(
        test_data_dict, node_feature_num, test_preprocessed_path
    )


def get_lowest_index(value: float, vector: np.ndarray) -> int:
    """Finds the index of the first element in a vector that is greater than the given value.

    Args:
        value: The value to compare.
        vector: The vector to search in.

    Returns:
        The index of the first element greater than the value.
    """
    for i, vec_value in enumerate(vector):
        if value < vec_value:
            return i + 1
    return len(vector)
