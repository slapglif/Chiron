# chiron/layers/snn/graph_attention.py

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation.

    This layer applies a multi-head attention mechanism to the input tensor,
    allowing nodes in the graph to attend to each other based on their feature
    representations and the graph structure.

    Args:
        in_features (int): Number of input features for each node.
        out_features (int): Number of output features for each node.
        num_heads (int): Number of attention heads. Default is 1.
        dropout (float): Dropout probability. Default is 0.0.
        alpha (float): Negative slope for the LeakyReLU activation function. Default is 0.2.
        concat (bool): If True, concatenate the output of all attention heads. Otherwise, average them. Default is True.

    Attributes:
        W (nn.Parameter): Linear transformation weight matrix for computing attention scores.
        a (nn.Parameter): Learnable attention mechanism coefficient.
        leakyrelu (nn.LeakyReLU): LeakyReLU activation function.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int = 1,
            dropout: float = 0.0,
            alpha: float = 0.2,
            concat: bool = True,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat

        # Calculate the number of output features per attention head
        self.out_features_per_head = out_features // num_heads

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

        # Reshape the input tensor to (batch_size, seq_len, num_heads, out_features_per_head)
        input_tensor = input_tensor.view(batch_size, seq_len, self.num_heads, -1)

        # Compute the attention scores using the attention mechanism coefficient
        attn_scores = torch.einsum(
            "bihk,bjhk->bhij", input_tensor, input_tensor
        )
        attn_scores = self.leakyrelu(attn_scores)

        return attn_scores

    def forward(
            self,
            input_tensor: torch.Tensor,
            adj_matrix: Optional[torch.sparse.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the output features for the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            adj_matrix (torch.sparse.Tensor, optional): Adjacency matrix tensor of shape (seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, num_heads * out_features) or (batch_size, seq_len, out_features).
        """
        batch_size, seq_len, _ = input_tensor.size()

        # Compute attention scores
        attn_scores = self._compute_attention_scores(input_tensor)

        logger.debug(f"input_tensor shape: {input_tensor.size()}")
        logger.debug(f"attn_scores shape: {attn_scores.shape}")

        # Apply the adjacency matrix mask to the attention scores (if provided)
        if adj_matrix is not None:
            num_nodes = adj_matrix.size(0)
            if num_nodes < seq_len:
                # If the adjacency matrix has fewer nodes than the sequence length,
                # pad the adjacency matrix with identity connections for the remaining nodes.
                indices = adj_matrix._indices()
                values = adj_matrix._values()
                indices = indices.clamp(max=seq_len - 1)
                adj_matrix = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=(seq_len, seq_len),
                    device=adj_matrix.device,
                )
            elif num_nodes > seq_len:
                # If the adjacency matrix has more nodes than the sequence length,
                # clip the adjacency matrix to the sequence length.
                indices = adj_matrix._indices()
                values = adj_matrix._values()
                mask = (indices[0] < seq_len) & (indices[1] < seq_len)
                indices = indices[:, mask]
                values = values[mask]
                adj_matrix = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=(seq_len, seq_len),
                    device=adj_matrix.device,
                )

            edge_indices = adj_matrix._indices()
            logger.debug(f"edge_indices: {edge_indices}")
            logger.debug(f"edge_indices shape: {edge_indices.shape}")

            if edge_indices.shape[-1] == 0:
                logger.warning(
                    "edge_indices tensor has an invalid shape along the last dimension."
                )
                mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            else:
                edge_indices_batch = edge_indices.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                edge_indices_head = edge_indices_batch.unsqueeze(1).expand(
                    -1, self.num_heads, -1, -1
                )

                logger.debug(f"edge_indices_head: {edge_indices_head}")
                logger.debug(f"edge_indices_head shape: {edge_indices_head.shape}")

                # Create the mask tensor with the correct shape
                mask_shape = (
                    batch_size,
                    self.num_heads,
                    attn_scores.size(2),
                    edge_indices_head.size(-1),
                )
                mask = torch.zeros(
                    mask_shape, dtype=torch.bool, device=attn_scores.device
                )
                mask.scatter_(3, edge_indices_head, True)

                logger.debug(f"mask shape after scatter: {mask.shape}")

                # Check if the mask shape matches the attn_scores shape along the desired dimensions
                expected_mask_shape = mask.shape
                if attn_scores.shape != expected_mask_shape:
                    raise ValueError(
                        f"Attention scores shape {attn_scores.shape} does not match the expected mask shape {expected_mask_shape}"
                    )

            # Apply the mask to the attention scores
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Apply softmax to normalize the attention scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Reshape the input tensor to (batch_size, seq_len, num_heads, out_features_per_head)
        input_tensor = input_tensor.view(batch_size, seq_len, self.num_heads, -1)

        # Apply attention probabilities to the input tensor
        attn_output = torch.einsum("bhij,bjhk->bihk", attn_probs, input_tensor)

        # Reshape the attention output
        if self.concat:
            attn_output = attn_output.reshape(
                batch_size, seq_len, self.num_heads * attn_output.size(-1)
            )
        else:
            attn_output = attn_output.mean(dim=1)

        return attn_output
