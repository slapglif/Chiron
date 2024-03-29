import torch
import torch.nn as nn


class SpikingTransformerEncoder(nn.Module):
    """
    Spiking Transformer Encoder module.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        """
        Initialize the SpikingTransformerEncoder.

        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            dropout (float): Dropout probability.
        """
        super(SpikingTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.device = torch.device("cuda:0")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=True
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the SpikingTransformerEncoder.

        Args:
            src (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the transformer encoder.
        """
        src = src.view(-1, 1)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the PositionalEncoding.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout probability. Default is 0.1.
            max_len (int): Maximum sequence length. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the PositionalEncoding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Positionally encoded tensor.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
