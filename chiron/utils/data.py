from typing import Tuple, Optional

import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SemanticFoldingDataset(Dataset):
    """
    Dataset class for Semantic Folding models.

    Instead of converting SDR embeddings to space-separated strings and re-tokenizing
    with a BERT tokenizer (which is semantically meaningless and lossy), this
    implementation directly quantizes the SDR embedding values into discrete token IDs
    using uniform quantization across the tokenizer's vocabulary.

    Features:
        - Uniform quantization: maps each float value to one of vocab_size bins
        - Proper attention mask: 1 for real tokens, 0 for padding
        - Data augmentation: random masking of SDR bits during training
    """

    def __init__(
        self,
        sdr_embeddings: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        labels: torch.Tensor = None,
        max_seq_len: int = 10,
        augment: bool = False,
        mask_prob: float = 0.15,
    ):
        """
        Initialize the SemanticFoldingDataset.

        Args:
            sdr_embeddings (torch.Tensor): The SDR embeddings.
            tokenizer (PreTrainedTokenizer): The tokenizer (used to determine vocab_size
                and special token IDs for quantization).
            labels (torch.Tensor, optional): The labels for the dataset. Default is None.
            max_seq_len (int): The maximum sequence length for tokenization.
            augment (bool): Whether to apply data augmentation (random masking) during
                training. Default is False.
            mask_prob (float): Probability of masking each SDR token during augmentation.
                Default is 0.15.
        """
        self.sdr_embeddings = sdr_embeddings
        self.tokenizer = tokenizer
        self.labels = (
            labels
            if labels is not None
            else torch.zeros((len(sdr_embeddings), max_seq_len), dtype=torch.long)
        )
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.mask_prob = mask_prob

        # Quantization parameters
        self.vocab_size = tokenizer.vocab_size
        # Reserve special token IDs: 0 = [PAD], so we map into range [1, vocab_size - 1]
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.mask_token_id = (
            tokenizer.mask_token_id
            if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None
            else 103  # Default BERT [MASK] token ID
        )
        # Number of bins for quantization (excluding pad token)
        self._num_bins = self.vocab_size - 1  # bins mapped to IDs [1, vocab_size - 1]

        # Pre-compute global min/max for uniform quantization
        self._global_min = self.sdr_embeddings.min().item()
        self._global_max = self.sdr_embeddings.max().item()
        # Avoid division by zero if all values are identical
        self._range = self._global_max - self._global_min
        if self._range == 0:
            self._range = 1.0

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.sdr_embeddings)

    def _quantize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Quantize a continuous SDR embedding into discrete token IDs using uniform
        quantization. Each float value is mapped to one of vocab_size bins.

        Args:
            embedding (torch.Tensor): A 1-D tensor of float SDR values.

        Returns:
            torch.Tensor: A 1-D tensor of integer token IDs in [1, vocab_size - 1].
        """
        # Normalize to [0, 1]
        normalized = (embedding.float() - self._global_min) / self._range
        # Clamp to handle any numerical edge cases
        normalized = normalized.clamp(0.0, 1.0)
        # Map to bin indices in [0, num_bins - 1], then shift to [1, num_bins]
        token_ids = (normalized * (self._num_bins - 1)).long() + 1
        return token_ids

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - input_ids: The quantized token IDs (long tensor of shape [max_seq_len]).
                - attention_mask: The attention mask (1 for real tokens, 0 for padding).
                - label: The label for the sample.
                - node_index: The index of the node.
        """
        sdr_embedding = self.sdr_embeddings[index]
        label = self.labels[index]
        node_index = index

        # Quantize SDR embedding values directly into token IDs
        input_ids = self._quantize_embedding(sdr_embedding)

        # Truncate or pad to max_seq_len
        seq_len = input_ids.shape[0]

        if seq_len >= self.max_seq_len:
            # Truncate
            input_ids = input_ids[: self.max_seq_len]
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        else:
            # Pad with pad_token_id
            padding_length = self.max_seq_len - seq_len
            input_ids = torch.cat(
                [input_ids, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)]
            )
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long),
                ]
            )

        # Data augmentation: random masking of SDR tokens during training
        if self.augment and self.training_mode:
            mask = torch.bernoulli(
                torch.full(input_ids.shape, self.mask_prob)
            ).bool()
            # Only mask real tokens (not padding)
            mask = mask & (attention_mask == 1)
            input_ids = input_ids.clone()
            input_ids[mask] = self.mask_token_id

        # Ensure the label has the same sequence length as input_ids
        label_padding_length = self.max_seq_len - label.shape[0]
        if label_padding_length > 0:
            label = torch.cat(
                [label, torch.full((label_padding_length,), -100, dtype=torch.long)]
            )
        else:
            label = label[: self.max_seq_len]

        return input_ids, attention_mask, label, node_index

    @property
    def training_mode(self) -> bool:
        """Whether the dataset is in training mode (enables augmentation).

        This checks the `augment` flag and can be toggled externally.
        Returns True by default when `augment` is set.
        """
        return getattr(self, "_training", True)

    @training_mode.setter
    def training_mode(self, value: bool) -> None:
        """Set the training mode flag."""
        self._training = value
