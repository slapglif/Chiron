# model.py

import random
from typing import List

import hnswlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from tqdm import tqdm

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer
from chiron.layers.sst.decoder import SpikingTransformerDecoder
from chiron.layers.sst.encoder import SpikingTransformerEncoder


def create_adjacency_matrix(sdr_embeddings: List[List[float]], threshold: float = 0.5,
                            subsample_rate: float = 0.1) -> coo_matrix:
    num_nodes = len(sdr_embeddings)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in tqdm(range(num_nodes), desc="Creating matrix..."):
        for j in range(i + 1, num_nodes):
            if random.random() < subsample_rate:
                similarity = 1 - cosine(sdr_embeddings[i], sdr_embeddings[j])
                if similarity >= threshold:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

    return coo_matrix(adjacency_matrix)


def create_adjacency_matrix_parallel(
        sdr_embeddings,
        n_jobs=-1,
        chunk_size=2000,
        threshold=0.5,
        subsample_rate=0.1,
        ef_construction=4000,
        m=256
):
    n = sdr_embeddings.shape[0]
    d = sdr_embeddings.shape[1]

    # Initialize HNSW index
    index = hnswlib.Index(space='cosine', dim=d)
    index.init_index(max_elements=2 * n, ef_construction=ef_construction, M=m)

    # Add items to the index in chunks
    num_chunks = (n + chunk_size - 1) // chunk_size
    for i in tqdm(range(num_chunks), desc="Adding items to index"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        index.add_items(sdr_embeddings[start:end], ids=np.arange(start, end))

    # Create sparse matrix
    rows, cols, data = [], [], []
    k = 100  # Adjust the value of k to a smaller fixed value
    for i in tqdm(range(num_chunks), desc="Building adjacency matrix"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        labels, distances = index.knn_query(
            sdr_embeddings[start:end], k=k, num_threads=-1
        )
        for j in range(end - start):
            for ki, dist in zip(labels[j], distances[j]):
                if start + j != ki and dist >= threshold and np.random.rand() < subsample_rate:
                    rows.append(start + j)
                    cols.append(ki)
                    data.append(1)

    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return adjacency_matrix


class SNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, timesteps: int, num_nodes: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.size()

        if self.mem1 is None:
            self.mem1 = torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device)
            self.mem2 = torch.zeros(batch_size, num_nodes, self.output_size).to(x.device)
            self.spike1 = torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device)
            self.spike2 = torch.zeros(batch_size, num_nodes, self.output_size).to(x.device)

        spikes = []

        for _ in range(self.timesteps):
            curr1 = self.fc1(x)
            curr1 = curr1.view(batch_size, num_nodes, self.hidden_size)  # Reshape curr1
            self.mem1 += curr1
            self.spike1.masked_fill_(self.mem1 > 0.5, 1.0)
            self.mem1.masked_fill_(self.mem1 > 0.5, 0.0)

            curr2 = self.fc2(self.spike1)
            self.mem2 += curr2
            self.spike2.masked_fill_(self.mem2 > 0.5, 1.0)
            self.mem2.masked_fill_(self.mem2 > 0.5, 0.0)

            spikes.append(self.spike2)

        spikes = torch.stack(spikes, dim=1)
        return spikes


class SNNModel(nn.Module):
    def __init__(self, sp_params: dict, snn_params: dict, gat_params: dict,
                 encoder_params: dict, decoder_params: dict, htm_params: dict):
        """
        Initialize the SNNModel.

        Args:
            sp_params (dict): Spatial pooler parameters.
            snn_params (dict): SNN layer parameters.
            gat_params (dict): Graph attention layer parameters.
            encoder_params (dict): Encoder parameters.
            decoder_params (dict): Decoder parameters.
            htm_params (dict): HTM layer parameters.
        """
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.snn_params = snn_params
        self.gat_params = gat_params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.htm_params = htm_params

        self.snn_layer = SNNLayer(**snn_params)
        self.gat_layer = GraphAttentionLayer(gat_params['in_features'], gat_params['out_features'])

        encoder_layer = nn.TransformerEncoderLayer(
            encoder_params['d_model'],
            encoder_params['nhead'],
            dropout=encoder_params['dropout'],
            batch_first=True,
            norm_first=True,
            dim_feedforward=2048,
            activation=F.gelu,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            encoder_params['num_layers'],
            enable_nested_tensor=True,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            decoder_params['d_model'],
            decoder_params['nhead'],
            dropout=decoder_params['dropout'],
            batch_first=True,
            norm_first=True,
            dim_feedforward=2048,
            activation=F.gelu,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            decoder_params['num_layers'],
        )

        self.htm_layer = HTMModel(**htm_params)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, conversation_texts: List[str]) -> torch.Tensor:
        """
        Perform forward pass of the SNNModel.

        Args:
            x (torch.Tensor): Input features.
            adj_matrix (torch.Tensor): Adjacency matrix.
            conversation_texts (List[str]): List of conversation texts.

        Returns:
            torch.Tensor: Output of the model.
        """
        x = x.unsqueeze(1)
        snn_output = self.snn_layer(x)

        batch_size, timesteps, num_nodes, _ = snn_output.size()
        snn_output = rearrange(snn_output, 'b t n d -> b (t n) d')

        # Preprocess conversation texts
        conversation_texts_str = [' '.join(conversation) for conversation in conversation_texts]

        gat_output = self.gat_layer(snn_output, adj_matrix, conversation_texts_str)

        gat_output = rearrange(gat_output, 'b (n t) d -> (b n) t d', n=num_nodes)
        encoder_output = self.encoder(gat_output)

        encoder_output = rearrange(encoder_output, '(b t) n d -> (b n) t d', b=batch_size)
        decoder_output = self.decoder(encoder_output, encoder_output)

        decoder_output = rearrange(decoder_output, '(b n) t d -> b n t d', b=batch_size)
        htm_output = self.htm_layer(decoder_output)

        return htm_output