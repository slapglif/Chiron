import concurrent.futures
import random
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedTokenizer

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
                [input_ids, torch.zeros(input_ids.size(0), padding_size, device=self.device)], dim=-1
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


class SNNModel(nn.Module):
    """
    Spiking Neural Network model integrating HTM and Graph Attention layers.

    Args:
        sp_params (dict): Parameters for the spatial pooling layer.
        snn_params (dict): Parameters for the SNN layer.
        gat_params (dict): Parameters for the Graph Attention layer.
        htm_params (dict): Parameters for the HTM layer.
        device (torch.device): The device to run the model on.
        vocab (dict): The vocabulary mapping for token to index.

    Attributes:
        sp_params (dict): Spatial pooling parameters.
        snn_params (dict): SNN layer parameters.
        gat_params (dict): Graph Attention layer parameters.
        htm_params (dict): HTM layer parameters.
        device (torch.device): Computation device.
        vocab (dict): Vocabulary mapping.
        output_size (int): Output size derived from `sp_params`.
        snn_layer (SNNLayer): The SNN layer of the model.
        gat_layer (GraphAttentionLayer): The GAT layer of the model.
        htm_layer (HTMModel): The HTM layer of the model.
        fc_out (nn.Linear): Final fully connected layer.
    """

    def __init__(
        self,
        sp_params: dict,
        snn_params: dict,
        gat_params: dict,
        htm_params: dict,
        device: torch.device,
        vocab: dict,
        tokenizer: PreTrainedTokenizer,
        max_sequence_length: int = 1000000,
    ):
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.snn_params = snn_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_sequence_length = max_sequence_length

        self.output_size = sp_params["sdr_dimensions"]
        self.snn_layer = SNNLayer(
            input_size=max_sequence_length,
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
            f"SNNModel initialized with sp_params={sp_params}, snn_params={snn_params},"
            f"gat_params={gat_params}, htm_params={htm_params}, output_size={self.output_size}"
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
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size).
            adj_matrix (torch.sparse_coo_tensor): Adjacency matrix in sparse COO format.
            node_indices (Optional[torch.Tensor]): Tensor of node indices. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        if input_ids.ndim == 2:
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
        if adj_matrix is not None:
            logger.debug(f"Adjacency matrix shape: {adj_matrix.shape}")
        if node_indices is not None:
            logger.debug(f"Node indices shape: {node_indices.shape}")

        # Forward pass through the SNN layer
        snn_output = self.snn_layer(input_ids, attention_mask, adj_matrix, node_indices)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Reshape the SNN output tensor
        snn_output = snn_output.view(snn_output.size(0), 1, -1)
        logger.debug(f"Reshaped SNN output shape: {snn_output.shape}")

        # Forward pass through the GAT layer
        gat_output = self.gat_layer(snn_output)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Forward pass through the HTM layer
        htm_output = self.htm_layer(gat_output).view(batch_size, -1)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Final fully connected layer
        output = self.fc_out(htm_output)
        logger.debug(f"Final output shape: {output.shape}")

        return output

    def generate(
        self,
        input_conversation: List[str],
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        adjacency_matrix_sparse: Optional[torch.Tensor] = None,
    ) -> List[List[str]]:
        """
        Generate responses based on the input conversation.

        Args:
            input_conversation (List[str]): The input conversation as a list of strings.
            max_length (int): The maximum length of the generated response. Default is 100.
            num_return_sequences (int): The number of responses to generate. Default is 1.
            temperature (float): The temperature for sampling. Default is 0.7.
            top_k (int): The number of top-k tokens to consider for filtering. Default is 50.
            top_p (float): The cumulative probability threshold for filtering. Default is 0.9.
            mirostat_eta (float): The learning rate for Mirostat. Default is 0.1.
            mirostat_tau (float): The target entropy for Mirostat. Default is 5.0.
            adjacency_matrix_sparse (Optional[torch.Tensor]): The adjacency matrix in sparse tensor format. Default is None.

        Returns:
            List[List[str]]: The list of generated conversations, where each conversation is a list of strings.
        """
        generated_conversations = []

        # Convert the input conversation to a list of strings
        string_conversation = [str(turn) for turn in input_conversation]

        # Check if the input conversation is empty
        if not string_conversation:
            logger.warning(
                "Input conversation is empty. Returning empty generated conversations."
            )
            return generated_conversations

        for _ in range(num_return_sequences):
            generated_conversation = []
            input_text = " ".join(string_conversation)
            generated_conversation.append(input_text)

            # Initialize Mirostat parameters
            mirostat_mu = 2.0 * mirostat_tau
            mirostat_s = 1.0

            # Prepare the input for the model
            model_input = self._prepare_input(generated_conversation)
            input_ids = model_input["input_ids"]
            attention_mask = model_input["attention_mask"]

            response = []
            while len(response) < max_length:
                # Forward pass
                model_output = self.forward(
                    input_ids,
                    attention_mask,
                    adjacency_matrix_sparse
                    if adjacency_matrix_sparse is not None
                    else torch.empty(0, dtype=torch.float32, device=self.device),
                    node_indices=None,
                )

                # Ensure model_output has the expected shape
                if model_output.dim() == 2:
                    model_output = model_output.unsqueeze(1)  # Add a sequence dimension

                # Sample from the model's output distribution
                output_logits = model_output[:, -1, :] / temperature
                filtered_logits = self._top_k_top_p_filtering(
                    output_logits, top_k=top_k, top_p=top_p
                )
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Update Mirostat parameters
                mirostat_s = (
                    mirostat_eta * (output_logits.max().item() - mirostat_tau)
                    + (1 - mirostat_eta) * mirostat_s
                )
                mirostat_mu = mirostat_mu * torch.exp(mirostat_s)
                temperature = max(0.1, temperature * (mirostat_tau / mirostat_mu))

                if next_token.item() == self.eos_token_id:
                    break

                response.append(next_token.item())
                model_input = self._prepare_input(
                    generated_conversation + [self.tokenizer.decode(response)]
                )
                input_ids = model_input["input_ids"]
                attention_mask = model_input["attention_mask"]

            generated_response = self.tokenizer.decode(response)
            generated_conversation.append(generated_response)
            generated_conversations.append(generated_conversation)

        return generated_conversations

    def _prepare_input(self, conversation: list) -> dict:
        """
        Prepare the input for the model based on the conversation structure.

        Args:
            conversation (list): The conversation as a list of strings.

        Returns:
            dict: The prepared input for the model.
        """
        input_ids = []
        attention_mask = []

        for turn in conversation:
            if isinstance(turn, str):
                encoded_turn = self.tokenizer.encode(turn, add_special_tokens=False)
                if encoded_turn:
                    input_ids.extend(encoded_turn + [self.tokenizer.eos_token_id])
                    attention_mask.extend([1] * (len(encoded_turn) + 1))
            elif turn is None:
                continue
            else:
                raise ValueError(f"Unexpected type for turn: {type(turn)}")

        # Remove None values from input_ids
        input_ids = [token for token in input_ids if token is not None]

        # Truncate input_ids and attention_mask if they exceed the maximum sequence length
        if len(input_ids) > self.max_sequence_length:
            input_ids = input_ids[: self.max_sequence_length]
            attention_mask = attention_mask[: self.max_sequence_length]

        if input_ids:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            logger.warning(
                "No valid turns found in the conversation. Returning empty input."
            )
            empty_tensor = torch.tensor([], dtype=torch.long)
            return {"input_ids": empty_tensor, "attention_mask": empty_tensor}

    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
    ) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

        Args:
            logits (torch.Tensor): Logits distribution shape (batch size, vocabulary size).
            top_k (int): Keep only top k tokens with highest probability (top-k filtering).
            top_p (float): Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            filter_value (float): Value to use for filtered logits.

        Returns:
            torch.Tensor: Filtered logits.
        """
        assert (
            logits.dim() == 1
        )  # Batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits
