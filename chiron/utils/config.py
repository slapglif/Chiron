import json
from typing import Any, Dict


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
            self.config: Dict[str, Any] = json.load(f)

    def __getitem__(self, key: str) -> Any:
        """
        Get the value of a configuration setting.

        Args:
            key (str): The key of the configuration setting.

        Returns:
            Any: The value of the configuration setting.
        """
        return self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get the value of a configuration setting with a default value.

        Args:
            key (str): The key of the configuration setting.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value of the configuration setting or the default value if the key is not found.
        """
        return self.config.get(key, default)

    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration setting exists.

        Args:
            key (str): The key of the configuration setting.

        Returns:
            bool: True if the configuration setting exists, False otherwise.
        """
        return key in self.config

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration settings with new values.

        Args:
            new_config (Dict[str, Any]): A dictionary of new configuration settings.
        """
        self.config.update(new_config)

    def save(self, config_path: str) -> None:
        """
        Save the configuration settings to a JSON file.

        Args:
            config_path (str): The path to save the configuration JSON file.
        """
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
