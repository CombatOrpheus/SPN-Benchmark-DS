import json
import os
from sklearn.model_selection import train_test_split
import numpy as np


def create_directory(path):
    """Creates a directory if it does not already exist.

    Args:
        path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


def load_data_from_txt(file_path):
    """Loads data from a comma-separated text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        numpy.ndarray: The data loaded from the text file.
    """
    return np.loadtxt(file_path, delimiter=",")


def load_json_file(file_path):
    """Loads data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def load_all_data_from_json_directory(directory_path):
    """Loads all JSON files from a directory and yields them one by one.

    Args:
        directory_path (str): The path to the directory containing the JSON files.

    Yields:
        dict: The data loaded from each JSON file.
    """
    json_files = sorted(os.listdir(directory_path), key=lambda x: int(x[4:-5]))
    for json_file in json_files:
        yield load_json_file(os.path.join(directory_path, json_file))


def count_json_files(directory_path):
    """Counts the number of JSON files in a directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        tuple: A tuple containing the number of JSON files and a list of their names.
    """
    json_files = sorted(os.listdir(directory_path), key=lambda x: int(x[4:-5]))
    return len(json_files), json_files


def load_all_data_from_txt_directory(directory_path):
    """Loads all text files from a directory into a list.

    Args:
        directory_path (str): The path to the directory containing the text files.

    Returns:
        list: A list of numpy.ndarray, each containing the data from a text file.
    """
    txt_files = sorted(os.listdir(directory_path), key=lambda x: int(x[4:-4]))
    all_data = []
    for txt_file in txt_files:
        data = load_data_from_txt(os.path.join(directory_path, txt_file))
        all_data.append(data)
    return all_data


def write_data_to_txt(file_path, data):
    """Writes data to a comma-separated text file.

    Args:
        file_path (str): The path to the text file.
        data (numpy.ndarray): The data to write.
    """
    np.savetxt(file_path, data, fmt="%.0f", delimiter=",")


def get_all_txt_files_in_subdirectories(directory_path):
    """Gets a list of all text files in the subdirectories of a given directory.

    Args:
        directory_path (str): The path to the main directory.

    Returns:
        list: A list of full paths to all text files.
    """
    subdirectories = sorted(os.listdir(directory_path), key=lambda x: int(x[4:]))
    all_txt_files = []
    for subdir in subdirectories:
        subdir_path = os.path.join(directory_path, subdir)
        txt_files = sorted(os.listdir(subdir_path), key=lambda x: int(x[4:-4]))
        full_paths = [os.path.join(subdir_path, f) for f in txt_files]
        all_txt_files.extend(full_paths)
    return all_txt_files


def load_arrivable_graph_data(directory_path, index):
    """Loads the data for an arrivable graph.

    Args:
        directory_path (str): The path to the directory containing the graph data.
        index (int): The index of the graph data to load.

    Returns:
        tuple: A tuple containing the node data, hu data, pb data, and transition number.
    """
    pb_data = load_data_from_txt(os.path.join(directory_path, f"pb_data{index}.txt")).astype(int)
    node_data = load_data_from_txt(os.path.join(directory_path, f"node_data{index}.txt")).astype(int)
    hu_data = load_data_from_txt(os.path.join(directory_path, f"hu_data{index}.txt")).astype(int)
    tran_num = load_data_from_txt(os.path.join(directory_path, f"tran_data{index}.txt")).astype(int)
    return node_data, hu_data, pb_data, tran_num


def save_data_to_json_file(file_path, data):
    """Saves data to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (dict): The data to save.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def get_all_txt_files_in_directory(directory_path):
    """Gets a list of all text files in a directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of text file names.
    """
    return sorted(os.listdir(directory_path), key=lambda x: int(x[4:-4]))


def add_preprocessed_to_dict(node_feature_num, key, value, preprocessed_dict):
    """Adds preprocessed data to a dictionary.

    Args:
        node_feature_num (int): The number of node features.
        key (str): The key for the new dictionary entry.
        value (dict): The original data dictionary.
        preprocessed_dict (dict): The dictionary to add the preprocessed data to.

    Returns:
        dict: The updated dictionary with the preprocessed data.
    """
    arrivable_dict = {}
    node_unit = []
    for row in value["arr_vlist"]:
        row_n = np.zeros(node_feature_num)
        row_n[:len(row)] = row
        node_unit.append(row_n.tolist())

    arrivable_dict["node_f"] = node_unit
    arrivable_dict["edge_index"] = value["arr_edge"]

    spn_lambda = np.array(value["spn_labda"])
    arr_tran_idx = [int(pa) for pa in value["arr_tranidx"]]
    arrivable_dict["edge_f"] = spn_lambda[arr_tran_idx].tolist()

    arrivable_dict["label"] = value["spn_mu"]
    preprocessed_dict[key] = arrivable_dict
    return preprocessed_dict


def sample_json_files_from_directory(num_samples, directory_path):
    """Samples a specified number of JSON files from a directory.

    Args:
        num_samples (int): The number of JSON files to sample.
        directory_path (str): The path to the directory.

    Returns:
        list: A list of dictionaries, each loaded from a sampled JSON file.
    """
    _, json_files = count_json_files(directory_path)
    sampled_files = np.random.choice(json_files, num_samples, replace=False)

    sampled_data = []
    for file_name in sampled_files:
        file_path = os.path.join(directory_path, file_name)
        data = load_json_file(file_path)
        # Filter out noisy data
        if -100 <= data["spn_mu"] <= 100:
            sampled_data.append(data)
        else:
            # Replace noisy data with a new random sample
            while True:
                new_file_name = np.random.choice(json_files, 1, replace=False)[0]
                if new_file_name not in sampled_files:
                    new_file_path = os.path.join(directory_path, new_file_name)
                    new_data = load_json_file(new_file_path)
                    if -100 <= new_data["spn_mu"] <= 100:
                        sampled_data.append(new_data)
                        sampled_files = np.append(sampled_files, new_file_name)
                        break
    return sampled_data


def create_data_dictionary(all_data):
    """Creates a dictionary from a list of data.

    Args:
        all_data (list): A list of data items.

    Returns:
        dict: A dictionary where keys are 'data1', 'data2', etc.
    """
    return {f"data{i+1}": data for i, data in enumerate(all_data)}


def load_arrivable_data_range(location_template, lower_limit, upper_limit):
    """Loads a range of data files based on a template.

    Args:
        location_template (str): A string template for the file path.
        lower_limit (int): The starting index.
        upper_limit (int): The ending index.

    Returns:
        list: A list of loaded data.
    """
    return [load_data_from_txt(location_template % str(i + 1)) for i in range(lower_limit, upper_limit)]


def load_lambda_mu_range(location_template, i, lower_limit, upper_limit):
    """Loads a range of lambda and mu data files.

    Args:
        location_template (str): A string template for the file path.
        i (int): The first index for the template.
        lower_limit (int): The starting second index.
        upper_limit (int): The ending second index.

    Returns:
        list: A list of loaded data.
    """
    return [load_data_from_txt(location_template % (str(i), str(j + 1))) for j in range(lower_limit, upper_limit)]


def _preprocess_and_save_data(data_dict, node_feature_num, file_path):
    """Preprocesses data and saves it to a JSON file."""
    preprocessed_dict = {}
    for key, value in data_dict.items():
        preprocessed_dict = add_preprocessed_to_dict(node_feature_num, key, value, preprocessed_dict)
    save_data_to_json_file(file_path, preprocessed_dict)


def partition_datasets(json_data_directory, node_feature_num, test_ratio=0.2):
    """Partitions the dataset into training and testing sets.

    Args:
        json_data_directory (str): The path to the directory with JSON data.
        node_feature_num (int): The number of node features.
        test_ratio (float, optional): The ratio of data to be used for testing. Defaults to 0.2.
    """
    original_data_dir = os.path.join(json_data_directory, "ori_data")
    preprocessed_data_dir = os.path.join(json_data_directory, "preprocessd_data")

    all_data_path = os.path.join(original_data_dir, "all_data.json")
    all_data = load_json_file(all_data_path)

    train_data, test_data = train_test_split(
        list(all_data.values()), test_size=test_ratio, random_state=0
    )

    train_data_dict = create_data_dictionary(train_data)
    test_data_dict = create_data_dictionary(test_data)

    save_data_to_json_file(os.path.join(original_data_dir, "train_data.json"), train_data_dict)
    save_data_to_json_file(os.path.join(original_data_dir, "test_data.json"), test_data_dict)

    create_directory(preprocessed_data_dir)

    train_preprocessed_path = os.path.join(preprocessed_data_dir, "train_data.json")
    _preprocess_and_save_data(train_data_dict, node_feature_num, train_preprocessed_path)

    test_preprocessed_path = os.path.join(preprocessed_data_dir, "test_data.json")
    _preprocess_and_save_data(test_data_dict, node_feature_num, test_preprocessed_path)


def get_lowest_index(value, vector):
    """Finds the index of the first element in a vector that is greater than the given value.

    Args:
        value: The value to compare.
        vector: The vector to search in.

    Returns:
        int: The index of the first element greater than the value.
    """
    for i, vec_value in enumerate(vector):
        if value < vec_value:
            return i + 1
    return len(vector)
