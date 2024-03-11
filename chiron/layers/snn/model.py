from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.distance import cosine

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer
from chiron.layers.sst.decoder import SpikingTransformerDecoder
from chiron.layers.sst.encoder import SpikingTransformerEncoder



def create_adjacency_matrix(
    input_features: List[List[float]], threshold: float = 0.5
) -> np.ndarray:
    """
    Create an adjacency matrix based on the similarity between input features.

    Args:
        input_features (List[List[float]]): List of input feature vectors.
        threshold (float): Similarity threshold for creating edges. Default is 0.5.

    Returns:
        np.ndarray: Adjacency matrix.
    """
    num_nodes = len(input_features)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = 1 - cosine(input_features[i], input_features[j])
            if similarity >= threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix


class SNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, timesteps: int,  num_nodes: int = 1):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.timesteps = timesteps

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.mem1 = None
        self.mem2 = None
        self.spike1 = None
        self.spike2 = None

        self.num_nodes = num_nodes  # Number of nodes in the layer
        self.device = torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_size = x.size()
        num_nodes = self.num_nodes

        x = x.view(batch_size, num_nodes, -1)  # Reshape x to (batch_size, num_nodes, input_size)

        if self.mem1 is None:
            self.mem1 = torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device)  # noqa: E501
            self.mem2 = torch.zeros(batch_size, num_nodes, self.output_size).to(x.device)  # noqa: E501
            self.spike1 = torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device)
            self.spike2 = torch.zeros(batch_size, num_nodes, self.output_size).to(x.device)

        spikes = []

        for _ in range(self.timesteps):
            curr1 = self.fc1(x)
            curr1 = curr1.view(batch_size, num_nodes, self.hidden_size)
            self.mem1 = self.mem1 + curr1
            self.spike1 = F.threshold(self.mem1, 0.5, 0)
            self.mem1 = self.mem1 - self.spike1

            curr2 = self.fc2(self.spike1)
            self.mem2 = self.mem2 + curr2
            self.spike2 = F.threshold(self.mem2, 0.5, 0)
            self.mem2 = self.mem2 - self.spike2

            spikes.append(self.spike2)

        spikes = torch.stack(spikes, dim=0)
        return spikes


class SNNModel(nn.Module):
    def __init__(
        self,
        sp_params: dict,
        snn_params: dict,
        gat_params: dict,
        encoder_params: dict,
        decoder_params: dict,
        htm_params: dict,
    ):
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.snn_params = snn_params
        self.gat_params = gat_params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.htm_params = htm_params

        self.snn_layer = SNNLayer(**snn_params)

        self.gat_layer = GraphAttentionLayer(**gat_params)

        self.encoder = SpikingTransformerEncoder(**encoder_params)


        self.decoder = SpikingTransformerDecoder(**decoder_params)
        self.htm_layer = HTMModel(**htm_params)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Apply SNN layer
        snn_output = self.snn_layer(x)

        # Apply GAT layer
        gat_output = self.gat_layer(snn_output[-1], adj_matrix)

        logger.info(f"GAT output shape: {gat_output.shape}")
        logger.info(f"Adj output shape {adj_matrix.shape}")
        # Apply Spiking Transformer Encoder
        encoder_output = self.encoder(gat_output)

        # Apply Spiking Transformer Decoder
        decoder_output = self.decoder(encoder_output)

        # Apply HTM layer
        htm_output = self.htm_layer(decoder_output)

        return htm_output

