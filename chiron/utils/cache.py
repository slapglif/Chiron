# chiron/utils/cache.py
from typing import Any

import numpy as np
import pyarrow as pa
import redis
from loguru import logger

redis_client = redis.Redis(host="localhost", port=6379, db=0)


def dict_to_struct_type(obj: dict) -> pa.DataType:
    """
    Convert a dictionary to a PyArrow StructType based on its keys and values.

    Args:
        obj (dict): The dictionary to convert.

    Returns:
        pa.DataType: The PyArrow StructType representation of the dictionary.
    """
    fields = []
    for key, value in obj.items():
        if isinstance(value, dict):
            field_type = dict_to_struct_type(value)
        elif isinstance(value, list):
            # Assume all list elements are of the same type as the first element
            element_type = (
                pa.float64()
                if all(isinstance(item, float) for item in value)
                else pa.string()
            )
            field_type = pa.list_(element_type)
        elif isinstance(value, float) or value is None:
            field_type = pa.float64()
        elif isinstance(value, int):
            field_type = pa.int64()  # Add this line to handle integer values
        else:
            field_type = pa.string()
        fields.append(pa.field(key, field_type))
    return pa.struct(fields)


def cache_data(data: Any, key: str) -> None:
    """
    Cache the data in Redis using Apache Arrow serialization.

    Args:
        data (Any): Data to be cached.
        key (str): Key to use for caching the data.
    """
    if isinstance(data, list) and len(data) == 0:
        logger.warning(f"Skipping caching for key '{key}' as data is an empty list.")
        return

    try:

        def convert_unsupported_types(item):
            if isinstance(item, tuple):
                return list(item)
            elif isinstance(item, (list, np.ndarray)):
                return [convert_unsupported_types(i) for i in item]
            elif isinstance(item, dict):
                return {k: convert_unsupported_types(v) for k, v in item.items()}
            else:
                return item

        data = convert_unsupported_types(data)

        # Convert data to a list of arrays if it's a list
        if isinstance(data, list):
            arrays = [pa.array(column) for column in zip(*data)]
        else:
            arrays = [pa.array(data)]

        # Create a RecordBatch from the arrays
        batch = pa.RecordBatch.from_arrays(
            arrays, names=[f"col{i}" for i in range(len(arrays))]
        )

        # Convert the RecordBatch to a Table
        table = pa.Table.from_batches([batch])

        # Serialize the Table
        serialized_data = table.to_string()
        # Store the serialized data in Redis
        redis_client.set(key, serialized_data)
        # logger.debug(f"Data cached with key: {key}")
    except Exception as e:
        logger.error(f"Error occurred while caching data for key '{key}': {str(e)}")


def load_cached_data(key: str) -> Any:
    # TODO: fix this method, it was breaking the app
    return None
    """
    Load cached data from Redis.

    Args:
        key (str): Key of the cached data.

    Returns:
        Any: Cached data if found, None otherwise.
    """
    serialized_data = redis_client.get(key)
    if serialized_data is not None:
        reader = pa.BufferReader(serialized_data)
        deserialized_data = pa.ipc.RecordBatchReader(reader)
        table = deserialized_data.read_all()
        if len(table.columns) == 1 and table.num_rows == 1:
            data = table.column(0).to_numpy()[0]
        else:
            data = [column.to_pylist() for column in table.columns]
            data = [list(row) for row in zip(*data)]
        return data


if __name__ == "__main__":
    example_data_simple = [
        {"name": "Alice", "age": 30, "interests": ["reading", "cycling"], "score": None}
    ]
    example_data_complex = [
        {
            "user": {
                "id": "user01",
                "preferences": {"likes": ["music", "art"], "dislikes": ["running"]},
            },
            "activity": {
                "type": "post",
                "content": "Loved the new art exhibit downtown!",
                "metrics": {"views": 150, "likes": 100, "shares": None},
            },
        }
    ]

# cache_data(example_data_simple, 'simple_data')
# cache_data(example_data_complex, 'complex_data')
# data = load_cached_data('simple_data')
# print(data)
# data = load_cached_data('complex_data')
# print(data)
