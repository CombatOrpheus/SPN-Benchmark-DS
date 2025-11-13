"""This module provides utility functions for writing data to files in various
formats, such as HDF5 and JSON-L. It includes a custom JSON encoder for
handling numpy data types.
"""

import json
from pathlib import Path
import numpy as np
import h5py
from typing import Any, Dict, IO


def write_to_hdf5(
    group: h5py.Group,
    data: Dict[str, Any],
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Writes a sample to an HDF5 group.

    Args:
        group: The HDF5 group to write to.
        data: The data to write.
        compression: The compression algorithm to use.
        compression_opts: The compression level.
    """
    for key, value in data.items():
        try:
            np_value = np.array(value)
            if np_value.ndim > 0:
                group.create_dataset(
                    key,
                    data=np_value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                group.create_dataset(key, data=np_value)

        except (TypeError, ValueError) as e:
            print(
                f"Warning: Could not save key '{key}' for sample "
                f"{group.name}. Error: {e}"
            )


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, o: Any) -> Any:
        """Handles numpy and Path objects for JSON serialization.

        Args:
            o: The object to encode.

        Returns:
            A serializable representation of the object.
        """
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_to_jsonl(file_handler: IO[str], data: Dict[str, Any]) -> None:
    """Appends a sample to a JSONL file, ensuring numpy compatibility.

    Args:
        file_handler: The file handler to write to.
        data: The data to write.
    """
    json_line = json.dumps(data, cls=NumpyEncoder)
    file_handler.write(f"{json_line}\n")
