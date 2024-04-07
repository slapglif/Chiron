import os
from typing import Union, Optional

import numpy as np
import pandas as pd
from loguru import logger


def cache_data(data: Union[pd.DataFrame, pd.Series, np.ndarray, list, dict], filepath: str) -> None:
    return
    """
    Cache data to a Feather file.

    Args:
        data: The data to be cached, which can be a pandas DataFrame or Series, a NumPy array, a list, or a dictionary.
        filepath: The path to the file where the data should be cached.
    """
    try:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_feather(filepath)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to pandas DataFrame first.
            pd.DataFrame(data).to_feather(filepath)
        elif isinstance(data, list) or isinstance(data, dict):
            # Convert list or dict to pandas DataFrame first.
            pd.DataFrame(data).to_feather(filepath)
        else:
            raise ValueError(f"Unsupported data type for caching: {type(data)}")
        logger.info(f"Data cached to file: {filepath}")
    except Exception as e:
        logger.error(f"Error occurred while caching data: {str(e)}")


def load_cached_data(filepath: str) -> Optional[Union[pd.DataFrame, pd.Series]]:
    return
    """
    Load cached data from a Feather file.

    Args:
        filepath: The path to the cached Feather file.

    Returns:
        The loaded data as a pandas DataFrame or Series, or None if an error occurs.
    """
    if os.path.exists(filepath):
        try:
            return pd.read_feather(filepath)
        except Exception as e:
            logger.error(f"Error occurred while loading data: {str(e)}")
    else:
        logger.warning(f"Cache file does not exist: {filepath}")
    return None
