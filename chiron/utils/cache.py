import os
import pickle
from typing import Union, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger


# Module-level LRU memory cache keyed by filepath.
# We use a wrapper since lru_cache requires hashable arguments.
_MEMORY_CACHE_MAX_SIZE = 128


class _MemoryCache:
    """In-memory LRU cache layer on top of the disk cache for fast repeated access."""

    def __init__(self, maxsize: int = _MEMORY_CACHE_MAX_SIZE):
        self._maxsize = maxsize
        self._cache: dict = {}
        self._access_order: list = []

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from memory cache, or None if not present."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Store an item in memory cache, evicting the least recently used if full."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict the least recently used entry
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        self._cache[key] = value
        self._access_order.append(key)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the memory cache. Returns True if the key was present."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear the entire memory cache."""
        self._cache.clear()
        self._access_order.clear()


# Singleton memory cache instance
_memory_cache = _MemoryCache()


def cache_data(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list, dict, Any], filepath: str
) -> None:
    """
    Cache data to a pickle file and update the in-memory LRU cache.

    Pickle is used instead of Feather for broader type support, including
    lists of numpy arrays, nested structures, and arbitrary Python objects.

    Args:
        data: The data to be cached. Supports pandas DataFrames/Series, NumPy arrays,
              lists, dicts, and any other picklable Python object.
        filepath: The path to the file where the data should be cached.

    Raises:
        TypeError: If the data is not picklable.
        OSError: If the file cannot be written (e.g., permission error, disk full).
    """
    try:
        # Ensure the parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Write to a temporary file first, then rename for atomicity
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename (on POSIX systems)
        os.replace(tmp_filepath, filepath)

        # Update memory cache
        _memory_cache.put(filepath, data)

        logger.info(f"Data cached to file: {filepath}")
    except (TypeError, pickle.PicklingError) as e:
        logger.error(f"Data is not picklable: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        raise TypeError(f"Cannot cache data of type {type(data)}: {str(e)}") from e
    except OSError as e:
        logger.error(f"OS error while caching data to {filepath}: {str(e)}")
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        raise
    except Exception as e:
        logger.error(f"Unexpected error while caching data: {str(e)}")
        if os.path.exists(filepath + ".tmp"):
            os.remove(filepath + ".tmp")


def load_cached_data(filepath: str) -> Optional[Any]:
    """
    Load cached data, checking the in-memory LRU cache first, then falling back
    to reading from the pickle file on disk.

    Args:
        filepath: The path to the cached pickle file.

    Returns:
        The loaded data, or None if the cache file does not exist or an error occurs.
    """
    # Check memory cache first for fast repeated access
    cached = _memory_cache.get(filepath)
    if cached is not None:
        logger.debug(f"Loaded data from memory cache: {filepath}")
        return cached

    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            # Store in memory cache for subsequent fast access
            _memory_cache.put(filepath, data)
            logger.info(f"Loaded cached data from disk: {filepath}")
            return data
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(
                f"Corrupted cache file {filepath}: {str(e)}. "
                "Invalidating cache entry."
            )
            invalidate_cache(filepath)
            return None
        except Exception as e:
            logger.error(f"Error loading cached data from {filepath}: {str(e)}")
            return None
    else:
        logger.warning(f"Cache file does not exist: {filepath}")
    return None


def invalidate_cache(filepath: str) -> None:
    """
    Invalidate a cache entry by removing it from both memory and disk.

    Args:
        filepath: The path to the cache file to invalidate.
    """
    # Remove from memory cache
    was_in_memory = _memory_cache.invalidate(filepath)

    # Remove from disk
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"Cache file invalidated and removed: {filepath}")
        except OSError as e:
            logger.error(f"Error removing cache file {filepath}: {str(e)}")
    elif was_in_memory:
        logger.info(f"Cache entry removed from memory (no disk file): {filepath}")
    else:
        logger.debug(f"No cache entry found to invalidate: {filepath}")


def clear_memory_cache() -> None:
    """
    Clear the entire in-memory LRU cache without touching disk files.
    Useful for freeing memory when the application is under memory pressure.
    """
    _memory_cache.clear()
    logger.info("In-memory cache cleared.")
