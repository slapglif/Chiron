import concurrent.futures
import random
from typing import List, Tuple
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


def process_chunk(
    sdr_embeddings: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    search_k: int,
    subsample_rate: float,
    threshold: float,
) -> List[Tuple[int, int]]:
    """
    Finds neighbors within a certain threshold for a chunk of data.

    Args:
        sdr_embeddings (np.ndarray): The SDR embeddings.
        chunk_start (int): The start index of the chunk.
        chunk_end (int): The end index of the chunk.
        search_k (int): Number of neighbors to consider.
        subsample_rate (float): Rate of subsampling to reduce connections.
        threshold (float): Threshold for considering two nodes as neighbors.

    Returns:
        List[Tuple[int, int]]: List of neighbor pairs.
    """
    chunk_adjacency_list = []
    index = faiss.IndexFlatL2(sdr_embeddings.shape[1])
    index.add(sdr_embeddings)

    for idx in range(chunk_start, chunk_end):
        _, neighbor_indices = index.search(
            np.expand_dims(sdr_embeddings[idx], axis=0), search_k
        )
        for neighbor_idx in neighbor_indices[0]:
            if idx != neighbor_idx and random.random() < subsample_rate:
                dist = np.linalg.norm(
                    sdr_embeddings[idx] - sdr_embeddings[neighbor_idx]
                )
                if dist < threshold:
                    chunk_adjacency_list.append((idx, neighbor_idx))

    return chunk_adjacency_list


def create_adjacency_matrix(
    sdr_embeddings: np.ndarray,
    threshold: float = 0.5,
    subsample_rate: float = 0.1,
    search_k: int = 1000,
    n_jobs: int = -1,
    chunk_size: int = 2500,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    """
    Creates a sparse adjacency matrix for the SDR embeddings using FAISS.

    Args:
        sdr_embeddings (np.ndarray): The SDR embeddings.
        threshold (float): Threshold for considering two nodes as neighbors.
        subsample_rate (float): Subsampling rate to reduce connections.
        search_k (int): Number of neighbors to consider.
        n_jobs (int): Number of parallel jobs to run.
        chunk_size (int): Size of each chunk to process in parallel.
        device (torch.device): The device to create the adjacency matrix on.

    Returns:
        torch.Tensor: The sparse adjacency matrix.
    """
    num_nodes = sdr_embeddings.shape[0]
    adjacency_list = []

    with tqdm(
        total=num_nodes, desc="Building adjacency matrix", unit="node"
    ) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for chunk_start in range(0, num_nodes, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_nodes)
                futures.append(
                    executor.submit(
                        process_chunk,
                        sdr_embeddings,
                        chunk_start,
                        chunk_end,
                        search_k,
                        subsample_rate,
                        threshold,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                adjacency_list.extend(future.result())
                progress_bar.update(chunk_size)

    rows, cols = zip(*adjacency_list)
    values = np.ones(len(rows), dtype=np.float32)
    adjacency_matrix_sparse = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(values),
        (num_nodes, num_nodes),
        device=device,
    )

    return adjacency_matrix_sparse


class SNNLayer(nn.Module):
    """
    Spiking Neural Network layer implementation.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        output_size (int): The number of output features.
        timesteps (int): The number of timesteps to unroll the layer for.
        num_nodes (int): The number of nodes in the graph.
        dropout (float): The dropout probability. Default is 0.0.
        device (torch.device): The device to run the computation on. Default is torch.device("cuda:0").
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        timesteps: int,
        num_nodes: int = 1,
        dropout: float = 0.0,
        device: torch.device = torch.device("cuda:0"),
    ):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.timesteps = timesteps
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.device = device

        # Initialize the fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, output_size).to(device)

        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"SNNLayer initialized with input_size={input_size}, hidden_size={hidden_size}, "
            f"output_size={output_size}, timesteps={timesteps}, num_nodes={num_nodes}, dropout={dropout}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrix: torch.sparse_coo_tensor,
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass of the SNNLayer.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size).
            adj_matrix (torch.sparse_coo_tensor): Adjacency matrix in sparse COO format.
            node_indices (Optional[torch.Tensor]): Tensor of node indices. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """

        # Ensure the input tensor has the correct shape
        if input_ids.size(-1) < self.input_size:
            # Pad input_ids with zeros to match the expected input_size
            padding_size = self.input_size - input_ids.size(-1)
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.zeros(input_ids.size(0), padding_size, device=self.device),
                ],
                dim=-1,
            )
        input_ids = input_ids.view(-1, self.input_size)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # Ensure input_ids has three dimensions [batch_size, seq_len, input_size]
        if len(input_ids.size()) == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)

        batch_size, seq_len, input_size = input_ids.size()

        if seq_len == 1:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

        assert seq_len == 1, "Input tensor should have a sequence length of 1"

        # Check if input_ids and attention_mask are empty tensors
        if input_ids.numel() == 0 and attention_mask.numel() == 0:
            logger.warning("Received empty input tensors. Returning zeros output.")
            return torch.zeros(batch_size, self.output_size, device=self.device)

        # Move the input tensors to the specified device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Ensure the input tensors are in float format
        input_ids = input_ids.float()
        attention_mask = attention_mask.float()

        logger.debug(f"Input tensor shape: {input_ids.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")
        logger.debug(f"Adjacency matrix shape: {adj_matrix.shape}")
        if node_indices is not None:
            logger.debug(f"Node indices shape: {node_indices.shape}")

        # Ensure the input tensor has the correct shape
        if input_ids.size(-1) != self.input_size:
            input_ids = input_ids.view(-1, self.input_size)

        # Apply the fully connected layers with dropout
        hidden_output = self.dropout_layer(torch.relu(self.fc1(input_ids)))
        output = self.fc2(hidden_output)
        logger.debug(f"SNN output shape: {output.shape}")

        return output


class LongformerLayer(nn.Module):
    """
    Longformer layer implementation for handling long sequences.

    Args:
        hidden_size (int): The hidden size of the Longformer model.
        num_layers (int): The number of Longformer layers.
        num_heads (int): The number of attention heads.
        max_length (int): The maximum sequence length.
        dropout (float): The dropout probability. Default is 0.1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        super(LongformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout = dropout

        # Initialize the Longformer model
        self.longformer = LongformerModel(
            config=LongformerConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                max_position_embeddings=max_length,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass of the LongformerLayer.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Ensure input_ids and attention_mask have the correct shape
        assert (
            input_ids.shape == attention_mask.shape
        ), "Input IDs and attention mask must have the same shape"

        # Forward pass through the Longformer model
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


class SNNModel(nn.Module):
    """
    Spiking Neural Network model integrating Longformer, SNNLayer, and Graph Attention layers.

    Args:
        sp_params (dict): Parameters for the spatial pooling layer.
        longformer_params (dict): Parameters for the Longformer layer.
        snn_params (dict): Parameters for the SNNLayer.
        gat_params (dict): Parameters for the Graph Attention layer.
        htm_params (dict): Parameters for the HTM layer.
        device (torch.device): The device to run the model on.
        vocab (dict): The vocabulary mapping for token to index.
        tokenizer (LongformerTokenizer): The Longformer tokenizer.
        max_sequence_length (int): The maximum sequence length.

    Attributes:
        sp_params (dict): Spatial pooling parameters.
        longformer_params (dict): Longformer layer parameters.
        snn_params (dict): SNNLayer parameters.
        gat_params (dict): Graph Attention layer parameters.
        htm_params (dict): HTM layer parameters.
        device (torch.device): Computation device.
        vocab (dict): Vocabulary mapping.
        tokenizer (LongformerTokenizer): Longformer tokenizer.
        max_sequence_length (int): Maximum sequence length.
        output_size (int): Output size derived from `sp_params`.
        longformer_layer (LongformerLayer): The Longformer layer of the model.
        snn_layer (SNNLayer): The SNNLayer of the model.
        gat_layer (GraphAttentionLayer): The GAT layer of the model.
        htm_layer (HTMModel): The HTM layer of the model.
        fc_out (nn.Linear): Final fully connected layer.
    """

    def __init__(
        self,
        sp_params: dict,
        longformer_params: dict,
        snn_params: dict,
        gat_params: dict,
        htm_params: dict,
        device: torch.device,
        vocab: dict,
        tokenizer: LongformerTokenizer,
        max_sequence_length: int,
    ):
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.longformer_params = longformer_params
        self.snn_params = snn_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self.output_size = sp_params["sdr_dimensions"]
        self.longformer_layer = LongformerLayer(
            hidden_size=longformer_params["hidden_size"],
            num_layers=longformer_params["num_layers"],
            num_heads=longformer_params["num_heads"],
            max_length=max_sequence_length,
            dropout=longformer_params.get("dropout", 0.1),
        ).to(device)
        self.snn_layer = SNNLayer(
            input_size=longformer_params["hidden_size"],
            hidden_size=snn_params["hidden_size"],
            output_size=snn_params["output_size"],
            timesteps=snn_params["timesteps"],
            num_nodes=self.output_size,
            dropout=snn_params.get("dropout", 0.0),
            device=device,
        ).to(device)
        self.gat_layer = GraphAttentionLayer(
            in_features=gat_params["in_features"],
            out_features=gat_params["out_features"],
            num_heads=gat_params["num_heads"],
            dropout=gat_params.get("dropout", 0.0),
            alpha=gat_params["alpha"],
            concat=gat_params["concat"],
        ).to(device)
        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)
        self.fc_out = nn.Linear(self.output_size, self.output_size).to(device)

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, longformer_params={longformer_params},"
            f"snn_params={snn_params}, gat_params={gat_params}, htm_params={htm_params}, output_size={self.output_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrix: torch.sparse_coo_tensor,
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass of the SNNModel.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            adj_matrix (torch.sparse_coo_tensor): Adjacency matrix in sparse COO format.
            node_indices (Optional[torch.Tensor]): Tensor of node indices. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Forward pass through the Longformer layer
        longformer_output = self.longformer_layer(input_ids, attention_mask)
        logger.debug(f"Longformer output shape: {longformer_output.shape}")

        # Implement chunking mechanism
        chunk_size = 512  # Adjust the chunk size based on available memory
        output_chunks = []

        for i in range(0, longformer_output.size(1), chunk_size):
            chunk_longformer_output = longformer_output[:, i : i + chunk_size, :]

            # Forward pass through the SNNLayer
            chunk_snn_output = self.snn_layer(
                chunk_longformer_output, None, adj_matrix, node_indices
            )
            logger.debug(f"SNNLayer output shape: {chunk_snn_output.shape}")

            # Reshape the SNNLayer output tensor
            chunk_snn_output = chunk_snn_output.view(chunk_snn_output.size(0), 1, -1)
            logger.debug(f"Reshaped SNNLayer output shape: {chunk_snn_output.shape}")

            output_chunks.append(chunk_snn_output)

        snn_output = torch.cat(output_chunks, dim=1)
        logger.debug(f"Concatenated SNNLayer output shape: {snn_output.shape}")

        # Forward pass through the GAT layer
        gat_output = self.gat_layer(snn_output)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Forward pass through the HTM layer
        htm_output = self.htm_layer(gat_output).view(gat_output.size(0), -1)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Final fully connected layer
        output = self.fc_out(htm_output)
        logger.debug(f"Final output shape: {output.shape}")

        return output
