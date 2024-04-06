import os
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger


def cache_data(data: Any, filepath: str) -> None:
    try:
        if isinstance(data, list):
            data = pa.array(data)
        elif isinstance(data, np.ndarray):
            data = pa.array(data)
        elif isinstance(data, dict):
            data = pa.Table.from_pydict(mapping=data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Write the data to a Feather file
        pa.feather.write_feather(data, filepath)
        logger.debug(f"Data cached to file: {filepath}")
    except Exception as e:
        logger.error(f"Error occurred while caching data to file '{filepath}': {str(e)}")


def load_cached_data(filepath: str) -> Any:
    if os.path.exists(filepath):
        try:
            # Read the Feather file
            data = pa.feather.read_feather(filepath)
            if isinstance(data, pa.Array):
                return data.to_pylist()
            elif isinstance(data, pa.Table):
                return data.to_pydict()
            else:
                return data
        except Exception as e:
            logger.error(f"Error loading cached data from file '{filepath}': {str(e)}")
    return None


if __name__ == "__main__":
    # Example usage
    example_data = [
        {"name": "Alice", "age": 30, "interests": ["reading", "cycling"], "score": None}
    ]

    cache_data(example_data, "example_data.feather")
    loaded_data = load_cached_data("example_data.feather")
    assert example_data == loaded_data, "Data caching/loading failed"
    logger.info("Data caching/loading passed")