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

        # Reshape the input tensor to (batch_size, seq_len, num_heads, out_features_per_head)
        input_tensor = input_tensor.view(batch_size, seq_len, self.num_heads, -1)

        # Compute the attention scores using the attention mechanism coefficient
        attn_scores = torch.einsum("bihk,bjhk->bhij", input_tensor, input_tensor)
        attn_scores = self.leakyrelu(attn_scores)

        return attn_scores

    def forward(
        self,
        input_tensor: torch.Tensor,
        adj_matrix: torch.sparse.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the output features for the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            adj_matrix (torch.sparse.Tensor, optional): Adjacency matrix tensor of shape (seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, num_heads * out_features_per_head) or (batch_size, seq_len, out_features).
        """
        batch_size, seq_len, num_features = input_tensor.size()

        # Compute attention scores
        attn_scores = self._compute_attention_scores(input_tensor)

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

            if edge_indices.shape[-1] == 0:
                logger.warning(
                    "edge_indices tensor has an invalid shape along the last dimension."
                )
                if self.fallback_mode == "dense":
                    # If fallback_mode is "dense", compute attention scores without the adjacency matrix mask
                    mask = None
                elif self.fallback_mode == "identity":
                    # If fallback_mode is "identity", apply an identity mask
                    mask_shape = (
                        batch_size,
                        self.num_heads,
                        seq_len,
                        seq_len,
                    )
                    mask = (
                        torch.eye(seq_len, device=attn_scores.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(batch_size, self.num_heads, 1, 1)
                    )
                else:
                    raise ValueError(f"Invalid fallback_mode: {self.fallback_mode}")
            else:
                # Create the mask tensor with the correct shape
                mask_shape = (
                    batch_size,
                    self.num_heads,
                    seq_len,
                    seq_len,
                )
                mask = torch.zeros(
                    mask_shape, dtype=torch.bool, device=attn_scores.device
                )
                mask[edge_indices[0], :, edge_indices[1], edge_indices[1]] = True

            # Apply the mask to the attention scores
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Apply softmax to normalize the attention scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

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
