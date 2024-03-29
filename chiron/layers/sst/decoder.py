import torch
import torch.nn as nn


class SpikingTransformerDecoder(nn.Module):
    """
    Spiking Transformer Decoder module.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        """
        Initialize the SpikingTransformerDecoder.

        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            dropout (float): Dropout probability.
        """
        super(SpikingTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, 1)
        self.device = torch.device("cuda:0")

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the SpikingTransformerDecoder.

        Args:
            tgt (torch.Tensor): Target tensor.
            memory (torch.Tensor): Memory tensor from the encoder.

        Returns:
            torch.Tensor: Output tensor from the transformer decoder.
        """
        tgt = self.embedding(tgt.unsqueeze(-1)).squeeze(1)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
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
