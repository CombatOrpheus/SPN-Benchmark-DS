"""
This module provides utility functions for writing data to files in various
formats, such as HDF5 and JSON-L. It includes a custom JSON encoder for
handling numpy data types.
"""

import json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm


def write_to_hdf5(group, data, compression="gzip", compression_opts=4):
    """Writes a sample to an HDF5 group."""
    for key, value in data.items():
        try:
            np_value = np.array(value)
            d_shape = np_value.shape
            d_type = np_value.dtype

            if np_value.ndim > 0:
                group.create_dataset(
                    key,
                    data=np_value,
                    shape=d_shape,
                    dtype=d_type,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                group.create_dataset(key, data=np_value, shape=d_shape, dtype=d_type)

        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save key '{key}' for sample {group.name}. Error: {e}")


def write_packed_hdf5(file_handle, samples_list, compression="gzip", compression_opts=4):
    """
    Writes a list of samples to HDF5 in a packed (columnar) format for efficient storage and retrieval.

    Args:
        file_handle (h5py.File or h5py.Group): The HDF5 file or group to write to.
        samples_list (list): A list of dictionaries containing the samples.
        compression (str): Compression filter.
        compression_opts (int): Compression level.
    """
    if not samples_list:
        return

    # Determine keys from the first sample
    # We assume all samples have the same keys.
    keys = list(samples_list[0].keys())

    # Iterate over each key to create columnar datasets
    for key in tqdm(keys, desc="Packing data"):
        # Collect values for this key across all samples
        values = [sample[key] for sample in samples_list]

        # Inspect the first value to determine type
        first_val = values[0]

        # Check if scalar (or 0-d array)
        is_scalar = False
        if isinstance(first_val, (int, float, bool, str, np.integer, np.floating, np.bool_)):
             is_scalar = True
        elif isinstance(first_val, (list, np.ndarray)):
             temp_arr = np.array(first_val)
             if temp_arr.ndim == 0:
                 is_scalar = True

        try:
            if is_scalar:
                # Store as a single dataset
                # Handle strings specially if needed, but h5py handles np.array of strings mostly
                # Convert to numpy array to ensure consistent dtype
                np_values = np.array(values)

                # Use string dtype for object arrays containing strings
                if np_values.dtype == object and len(np_values) > 0 and isinstance(np_values[0], str):
                    dt = h5py.string_dtype(encoding='utf-8')
                    file_handle.create_dataset(
                        key,
                        data=np_values,
                        dtype=dt,
                        compression=compression,
                        compression_opts=compression_opts
                    )
                else:
                    file_handle.create_dataset(
                        key,
                        data=np_values,
                        compression=compression,
                        compression_opts=compression_opts
                    )
            else:
                # Store as packed arrays: data, shapes, ptr
                flattened_data = []
                shapes = []
                lengths = []

                for v in values:
                    v_arr = np.array(v)
                    shapes.append(v_arr.shape)
                    flat = v_arr.flatten()
                    flattened_data.append(flat)
                    lengths.append(len(flat))

                if flattened_data:
                    concatenated = np.concatenate(flattened_data)
                else:
                    concatenated = np.array([])

                # Write data
                file_handle.create_dataset(
                    f"{key}_data",
                    data=concatenated,
                    compression=compression,
                    compression_opts=compression_opts
                )

                # Write shapes
                file_handle.create_dataset(
                    f"{key}_shapes",
                    data=np.array(shapes),
                    compression=compression,
                    compression_opts=compression_opts
                )

                # Write pointers (start indices)
                # ptr has length N+1
                ptr = np.zeros(len(values) + 1, dtype=np.int64)
                # cumsum of lengths gives the end indices
                # ptr[0] is 0
                # ptr[1] is length of first element, etc.
                np.cumsum(lengths, out=ptr[1:])

                file_handle.create_dataset(
                    f"{key}_ptr",
                    data=ptr,
                    compression=compression,
                    compression_opts=compression_opts
                )

        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save key '{key}' in packed format. Error: {e}")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def write_to_jsonl(file_handler, data):
    """Appends a sample to a JSONL file, ensuring numpy compatibility."""
    json_line = json.dumps(data, cls=NumpyEncoder)
    file_handler.write(json_line + "\n")
