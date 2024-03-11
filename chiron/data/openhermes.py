from typing import List, Dict


class OpenHermesDataset:
    """Class for loading and processing the OpenHermes dataset."""

    def __init__(self, config: Dict):
        """
        Initialize the OpenHermesDataset instance.

        Args:
            config (Dict): Configuration dictionary containing dataset parameters.
        """
        self.name = config["name"]
        self.path = config["path"]
        self.data = self.load_data()
        self.conversations = self.get_conversations()
        self.sources = self.get_sources()
        self.categories = self.get_categories()
        self.hashes = self.get_hashes()
        self.custom_instructions = self.get_custom_instructions()

    def load_data(self) -> List[Dict]:
        """
        Load the dataset from the JSON file.

        Returns:
            List[Dict]: List of dataset entries.
        """
        import json

        with open(self.path, "r") as file:
            data = json.load(file)

        return data

    def get_conversations(self) -> List[List[Dict]]:
        """
        Extract the conversations from the dataset.

        Returns:
            List[List[Dict]]: List of conversations, where each conversation is a list of turns.
        """
        conversations = [entry["conversations"] for entry in self.data]
        return conversations

    def get_sources(self) -> List[str]:
        """
        Extract the sources from the dataset.

        Returns:
            List[str]: List of sources.
        """
        sources = [entry.get("source", "") for entry in self.data]
        return sources

    def get_categories(self) -> List[str]:
        """
        Extract the categories from the dataset.

        Returns:
            List[str]: List of categories.
        """
        categories = [entry.get("category", "") for entry in self.data]
        return categories

    def get_hashes(self) -> List[str]:
        """
        Extract the hashes from the dataset.

        Returns:
            List[str]: List of hashes.
        """
        hashes = [entry.get("hash", "") for entry in self.data]
        return hashes

    def get_custom_instructions(self) -> List[bool]:
        """
        Extract the custom_instruction flags from the dataset.

        Returns:
            List[bool]: List of custom_instruction flags.
        """
        custom_instructions = [
            entry.get("custom_instruction", False) for entry in self.data
        ]  # noqa: E501
        return custom_instructions
