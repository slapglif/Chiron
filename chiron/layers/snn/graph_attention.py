from typing import Union

import numpy as np
import scipy
import torch
import torch.nn as nn
from loguru import logger


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
        fallback_mode: str = "dense",
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat
        self.fallback_mode = fallback_mode

        # Calculate the number of output features per attention head
        if concat:
            self.out_features_per_head = out_features // num_heads
        else:
            self.out_features_per_head = out_features

        # Initialize the linear transformation weight matrix
        self.W = nn.Parameter(
            torch.empty(size=(num_heads, in_features, self.out_features_per_head))
        )
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Initialize the attention mechanism coefficient
        self.a = nn.Parameter(
            torch.empty(size=(num_heads, 2 * self.out_features_per_head, 1))
        )
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Initialize the LeakyReLU activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def _compute_attention_scores(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores using the input features across all attention heads.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Attention scores of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, num_features = input_tensor.size()
        logger.debug(
            f"Input tensor shape in _compute_attention_scores: {input_tensor.shape}"
        )

        # Reshape the input tensor to (batch_size, seq_len, num_heads, out_features_per_head)
        input_tensor = input_tensor.view(batch_size, seq_len, self.num_heads, -1)
        logger.debug(
            f"Reshaped input tensor shape in _compute_attention_scores: {input_tensor.shape}"
        )

        # Compute the attention scores using the attention mechanism coefficient
        attn_scores = torch.einsum("bihk,bjhk->bhij", input_tensor, input_tensor)
        logger.debug(
            f"Attention scores shape before activation in _compute_attention_scores: {attn_scores.shape}"
        )
        attn_scores = self.leakyrelu(attn_scores)
        logger.debug(
            f"Attention scores shape after activation in _compute_attention_scores: {attn_scores.shape}"
        )

        return attn_scores

    def forward(
        self,
        input_tensor: torch.Tensor,
        adj_matrix: Union[np.ndarray, scipy.sparse.csr_matrix, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the output features for the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            adj_matrix (Union[np.ndarray, scipy.sparse.csr_matrix, torch.Tensor], optional): Adjacency matrix as a NumPy array, a SciPy sparse matrix, or a PyTorch tensor of shape (seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, num_heads * out_features_per_head) or (batch_size, seq_len, out_features).
        """
        batch_size, seq_len, num_features = input_tensor.size()
        logger.debug(f"input_tensor shape: {input_tensor.shape}")

        # Compute attention scores
        attn_scores = self._compute_attention_scores(input_tensor)
        logger.debug(f"attn_scores shape: {attn_scores.shape}")

        # Apply the adjacency matrix mask to the attention scores (if provided)
        if adj_matrix is not None:
            if isinstance(adj_matrix, np.ndarray):
                # Convert the adjacency matrix to a PyTorch tensor
                adj_matrix_tensor = torch.from_numpy(adj_matrix).to(input_tensor.device)
            elif isinstance(adj_matrix, scipy.sparse.csr_matrix):
                # Convert the sparse adjacency matrix to a dense PyTorch tensor
                adj_matrix_tensor = torch.from_numpy(adj_matrix.toarray()).to(
                    input_tensor.device
                )
            elif isinstance(adj_matrix, torch.Tensor):
                # Adjacency matrix is already a PyTorch tensor
                adj_matrix_tensor = adj_matrix.to(input_tensor.device)
            else:
                raise TypeError(f"Unsupported type for adj_matrix: {type(adj_matrix)}")

            # Ensure the adjacency matrix tensor has the correct shape
            seq_len, _ = adj_matrix_tensor.shape
            assert seq_len == input_tensor.size(
                1
            ), f"Adjacency matrix should have shape (seq_len, seq_len), but got {adj_matrix_tensor.shape}"

            # Expand the adjacency matrix tensor to match the shape of attn_scores
            adj_matrix_tensor = (
                adj_matrix_tensor.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_heads, seq_len, seq_len)
            )

            # Mask the attention scores with the adjacency matrix
            attn_scores = attn_scores.masked_fill(adj_matrix_tensor == 0, float("-inf"))

        # Apply softmax to normalize the attention scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        logger.debug(f"attn_probs shape: {attn_probs.shape}")

        # Reshape the input tensor to (batch_size, seq_len, num_heads, out_features_per_head)
        input_tensor = input_tensor.view(batch_size, seq_len, self.num_heads, -1)

        # Apply attention probabilities to the input tensor
        attn_output = torch.einsum("bhij,bjhk->bihk", attn_probs, input_tensor)

        if self.concat:
            # Reshape the attention output to (batch_size, seq_len, num_heads * out_features_per_head)
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
        else:
            # Average the attention output across the attention heads
            attn_output = attn_output.mean(dim=1)

        return attn_output
