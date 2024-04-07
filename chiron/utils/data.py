from typing import Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SemanticFoldingDataset(Dataset):
    """
    Dataset class for Semantic Folding models.
    """

    def __init__(self, sdr_embeddings, tokenizer: PreTrainedTokenizer, labels=None):
        """
        Initialize the SemanticFoldingDataset.

        Args:
            sdr_embeddings (np.ndarray): The SDR embeddings.
            tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
            labels (torch.Tensor, optional): The labels for the dataset. Default is None.
        """
        self.sdr_embeddings = sdr_embeddings
        self.tokenizer = tokenizer
        self.labels = (
            labels
            if labels is not None
            else torch.zeros((len(sdr_embeddings), sdr_embeddings.shape[1]))
        )

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.sdr_embeddings)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - input_ids: The tokenized input IDs.
                - attention_mask: The attention mask for the input.
                - label: The label for the sample.
                - node_index: The index of the node.
        """
        sdr_embedding = self.sdr_embeddings[index]
        label = self.labels[index]
        node_index = index

        # Tokenize the input using the tokenizer
        input_ids = self.tokenizer.encode(
            str(sdr_embedding),
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        input_ids = input_ids.squeeze(0)  # Remove the batch dimension
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, label.clone().detach(), node_index
