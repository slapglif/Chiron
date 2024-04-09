from typing import Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SemanticFoldingDataset(Dataset):
    """
    Dataset class for Semantic Folding models.
    """

    def __init__(
        self,
        sdr_embeddings: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        labels: torch.Tensor = None,
        max_seq_len: int = 10,
    ):
        """
        Initialize the SemanticFoldingDataset.

        Args:
            sdr_embeddings (torch.Tensor): The SDR embeddings.
            tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
            labels (torch.Tensor, optional): The labels for the dataset. Default is None.
            max_seq_len (int): The maximum sequence length for tokenization.
        """
        self.sdr_embeddings = sdr_embeddings
        self.tokenizer = tokenizer
        self.labels = (
            labels
            if labels is not None
            else torch.zeros((len(sdr_embeddings), max_seq_len), dtype=torch.long)
        )
        self.max_seq_len = max_seq_len

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

        # Convert the SDR embedding tensor to a string representation
        sdr_embedding_str = " ".join(map(str, sdr_embedding.tolist()))

        # Tokenize the input using the tokenizer
        input_ids = self.tokenizer.encode(
            sdr_embedding_str,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
        )
        input_ids = input_ids.squeeze(0)  # Remove the batch dimension

        # Create the attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Ensure the label has the same sequence length as input_ids
        label_padding_length = self.max_seq_len - label.shape[0]
        if label_padding_length > 0:
            label = torch.cat(
                [label, torch.full((label_padding_length,), -100, dtype=torch.long)]
            )
        else:
            label = label[: self.max_seq_len]

        return input_ids, attention_mask, label, node_index
