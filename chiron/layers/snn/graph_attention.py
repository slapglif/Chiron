import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation.

    This layer applies a multi-head attention mechanism to the input tensor,
    allowing nodes in the graph to attend to each other based on their feature
    representations and the graph structure.

    Args:
        in_features (int): Number of input features for each node.
        out_features (int): Number of output features for each node.
        num_heads (int, optional): Number of attention heads. Default is 1.
        dropout (float, optional): Dropout probability. Default is 0.0.
        alpha (float, optional): Negative slope for the LeakyReLU activation function. Default is 0.2.
        concat (bool, optional): If True, concatenate the output of all attention heads. Otherwise, average them. Default is True.

    Attributes:
        W (nn.Parameter): Linear transformation weight matrix for computing attention scores.
        a (nn.Parameter): Learnable attention mechanism coefficient.
        leakyrelu (nn.LeakyReLU): LeakyReLU activation function.
        dropout (nn.Dropout): Dropout layer.

    Example:
        >>> import torch
        >>> layer = GraphAttentionLayer(in_features=128, out_features=64, num_heads=2, dropout=0.1)
        >>> input_tensor = torch.randn(32, 10, 128)  # (batch_size, seq_len, num_features)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 10, 128])  # (batch_size, seq_len, num_heads * out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
    ) -> None:
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
        self.a = nn.Parameter(torch.empty(size=(2 * self.out_features_per_head, 1)))
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
        batch_size, seq_len, _ = input_tensor.size()

        # Reshape the input tensor to (batch_size, num_heads, seq_len, out_features_per_head)
        input_tensor = torch.einsum("bij,hjk->bihk", input_tensor, self.W)

        # Compute attention scores
        scores = torch.einsum("bihk,bjhk->bhij", input_tensor, input_tensor)
        scores = scores / (self.out_features_per_head**0.5)

        # Apply the LeakyReLU nonlinearity
        scores = self.leakyrelu(scores)

        return scores

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the output features for the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, num_heads * out_features) or (batch_size, seq_len, out_features).
        """
        batch_size, seq_len, _ = input_tensor.size()

        # Compute attention scores
        attn_scores = self._compute_attention_scores(input_tensor)

        # Apply attention scores to input features
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Reshape the input tensor to (batch_size, num_heads, seq_len, out_features_per_head)
        input_tensor = torch.einsum("bij,hjk->bihk", input_tensor, self.W)

        # Apply attention probabilities
        attn_output = torch.einsum("bhij,bihk->bjhk", attn_probs, input_tensor)

        # Reshape the output tensor
        if self.concat:
            attn_output = attn_output.reshape(
                batch_size, seq_len, self.num_heads * self.out_features_per_head
            )
        else:
            attn_output = attn_output.mean(dim=1)

        return attn_output
