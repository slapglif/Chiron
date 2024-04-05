from typing import Any
import numpy as np
import pyarrow as pa
import redis
import pandas as pd
from loguru import logger

redis_client = redis.Redis(host="localhost", port=6379, db=0)


def dict_to_struct_type(obj: dict) -> pa.DataType:
    fields = []
    for key, value in obj.items():
        if isinstance(value, dict):
            field_type = dict_to_struct_type(value)
        elif isinstance(value, list):
            element_type = (
                pa.float64()
                if all(isinstance(item, float) for item in value)
                else pa.string()
            )
            field_type = pa.list_(element_type)
        elif isinstance(value, float) or value is None:
            field_type = pa.float64()
        elif isinstance(value, int):
            field_type = pa.int64()
        else:
            field_type = pa.string()
        fields.append(pa.field(key, field_type))
    return pa.struct(fields)


from pyarrow import ipc


def cache_data(data: Any, key: str) -> None:
    return
    if isinstance(data, list) and len(data) == 0:
        logger.warning(f"Skipping caching for key '{key}' as data is an empty list.")
        return

    try:
        data = convert_unsupported_types(data)
        table = pa.Table.from_pandas(pd.DataFrame(data))

        # Serialize the Table using a RecordBatchStreamWriter
        buffer = pa.BufferOutputStream()
        with ipc.new_stream(buffer, table.schema) as writer:
            writer.write_table(table)
        serialized_data = buffer.getvalue()

        # Store the serialized data in Redis
        redis_client.set(key, serialized_data.to_pybytes())
        logger.debug(f"Data cached with key: {key}")
    except Exception as e:
        logger.error(f"Error occurred while caching data for key '{key}': {str(e)}")


def load_cached_data(key: str) -> Any:
    return
    serialized_data = redis_client.get(key)
    if serialized_data is not None:
        buffer = pa.py_buffer(serialized_data)
        try:
            # Use RecordBatchStreamReader for reading the stream
            reader = pa.ipc.RecordBatchStreamReader(buffer)
            table = reader.read_all()
            return table.to_pandas()
        except Exception as e:
            logger.error(f"Error loading cached data for key '{key}': {str(e)}")
    return None


# Debugging functions
def cache_data_debug(data: Any, filepath: str) -> None:
    try:
        table = pa.Table.from_pandas(pd.DataFrame(data))
        with pa.OSFile(filepath, "wb") as sink:
            with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
                writer.write_table(table)
        logger.debug(f"Data cached in file: {filepath}")
    except Exception as e:
        logger.error(f"Error caching data to file: {str(e)}")


def load_cached_data_debug(filepath: str) -> Any:
    try:
        with pa.OSFile(filepath, "rb") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()
        return table.to_pandas()
    except Exception as e:
        logger.error(f"Error loading data from file: {str(e)}")
        return None


def convert_unsupported_types(item):
    if isinstance(item, tuple):
        return list(item)
    elif isinstance(item, (list, np.ndarray)):
        return [convert_unsupported_types(i) for i in item]
    elif isinstance(item, dict):
        return {k: convert_unsupported_types(v) for k, v in item.items()}
    else:
        return item


if __name__ == "__main__":
    example_data = [
        {"name": "Alice", "age": 30, "interests": ["reading", "cycling"], "score": None}
    ]

    # Use Redis for caching
    cache_data(example_data, "example_data")
    data_from_redis = load_cached_data("example_data")
    logger.info(data_from_redis)

    # Use file for debugging
    cache_data_debug(example_data, "temp_data.arrow")
    data_from_file = load_cached_data_debug("temp_data.arrow")
    logger.info(data_from_file)
