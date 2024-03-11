import json
from typing import Any


class Config:
    """
    Configuration class to handle settings for the Semantic Folding pipeline.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Config instance.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def __getitem__(self, key: str) -> Any:
        """
        Get the value of a configuration setting.

        Args:
            key (str): The key of the configuration setting.

        Returns:
            Any: The value of the configuration setting.
        """
        return self.config[key]
