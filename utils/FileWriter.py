"""
This module provides utility functions for writing data to files in various
formats, such as HDF5 and JSON-L. It includes a custom JSON encoder for
handling numpy data types.
"""

import json
import numpy as np
import h5py


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


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def write_to_jsonl(file_handler, data):
    """Appends a sample to a JSONL file, ensuring numpy compatibility."""
    json_line = json.dumps(data, cls=NumpyEncoder)
    file_handler.write(json_line + "\n")
