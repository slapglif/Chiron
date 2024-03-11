import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        alpha: float,
        concat: bool = True,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the GraphAttentionLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            dropout (float): Dropout probability.
            alpha (float): LeakyReLU negative slope.
            concat (bool): Whether to concatenate or average the multi-head attention results. Default is True.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)  # Wh has shape (batch_size, num_nodes, out_features)

        # Compute attention scores
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # Apply mask to attention scores
        zero_vec = -9e15 * torch.ones_like(e)  # Modify the shape of zero_vec
        attention = torch.where(adj_mat > 0, e, zero_vec)

        # Normalize attention scores
        attention = F.softmax(attention, dim=1)

        # Reshape attention tensor
        attention = attention.unsqueeze(-1)  # Add an extra dimension to attention

        # Compute new node features
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare the input for the attention mechanism.

        Args:
            Wh (torch.Tensor): Linear transformed input node features of shape (batch_size, num_nodes, out_features).

        Returns:
            torch.Tensor: Input tensor for the attention mechanism of shape (batch_size, num_nodes, num_nodes, 2 * out_features).
        """
        batch_size, N, _ = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features)
