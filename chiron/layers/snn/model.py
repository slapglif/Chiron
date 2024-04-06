# chiron/layers/snn/model.py

from typing import Optional, Generator, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


class SNNLayer(nn.Module):
    """
    Spiking Neural Network (SNN) layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        timesteps: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the SNNLayer.

        Args:
            input_size (int): The size of the input tensor.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output tensor.
            timesteps (int): The number of timesteps for the spiking neurons.
            dropout (float): The dropout probability.
        """
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.timesteps = timesteps
        self.dropout = dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the SNN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).
        """
        # Ensure the input tensor is of the same type as the model's parameters
        x = x.to(dtype=self.fc2.weight.dtype, device=x.device)

        batch_size, seq_len, _ = x.shape

        # Reshape the input tensor to (batch_size, seq_len, input_size)
        x = x.view(batch_size, seq_len, self.input_size)

        mem1 = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype
        )
        mem2 = torch.zeros(
            batch_size, seq_len, self.output_size, device=x.device, dtype=x.dtype
        )

        spikes = []
        for _ in range(self.timesteps):
            mem1 = mem1 + self.fc1(x)
            spike1 = (mem1 > 0.5).float()  # Apply spiking behavior
            mem1 = mem1 * (mem1 <= 0.5)  # Reset membrane potential

            spike1 = self.dropout_layer(spike1)
            mem2 = mem2 + self.fc2(spike1)
            spike2 = (mem2 > 0.5).float()  # Apply spiking behavior
            mem2 = mem2 * (mem2 <= 0.5)  # Reset membrane potential

            spikes.append(spike2)

        # Stack the spikes along the timestep dimension
        output = torch.stack(spikes, dim=1)

        return output


class SNNModel(nn.Module):
    def __init__(
        self,
        sp_params: dict,
        gat_params: dict,
        htm_params: dict,
        device: torch.device,
        vocab: dict,
        tokenizer: PreTrainedTokenizer,
        snn_params: dict,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize the SNNModel.

        Args:
            sp_params (dict): Parameters for the Sparse Distributed Representation (SDR) generation.
            gat_params (dict): Parameters for the Graph Attention Network (GAT) layer.
            htm_params (dict): Parameters for the Hierarchical Temporal Memory (HTM) layer.
            device (torch.device): The device to use for model operations.
            vocab (dict): The vocabulary dictionary mapping tokens to indices.
            tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
            snn_params (dict): Parameters for the Spiking Neural Network (SNN) layer.
            batch_size (int): The batch size for processing input sequences.
            num_workers (int): The number of worker processes for data loading.
        """
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.output_size = sp_params["sdr_dimensions"]
        self.snn_params = snn_params
        self.concat = gat_params["concat"]

        self.snn_layer = None
        self.gat_layer = GraphAttentionLayer(
            in_features=self.snn_params["output_size"],
            out_features=self.gat_params["out_features"],
            num_heads=self.gat_params["num_heads"],
            alpha=self.gat_params["alpha"],
            concat=self.concat,
            fallback_mode=gat_params.get("fallback_mode", "dense"),
        ).to(device)

        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        if self.concat:
            self.fc_out = nn.Linear(
                self.gat_params["num_heads"] * self.gat_params["out_features"],
                self.output_size,
            ).to(device)
        else:
            self.fc_out = nn.Linear(
                self.gat_params["out_features"], self.output_size
            ).to(device)

        self.batch_size = batch_size
        self.num_workers = num_workers

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, gat_params={gat_params},"
            f"htm_params={htm_params}, output_size={self.output_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        adjacency_matrix_batches: Optional[Generator[torch.Tensor, None, None]] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the SNN model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            adjacency_matrix_batches (Optional[Generator[torch.Tensor, None, None]]): Generator of batch adjacency matrix tensors.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        device = self.device
        batch_size = self.batch_size
        num_workers = self.num_workers

        logger.debug(f"Input tensor shape: {input_ids.shape}")

        # Split the input sequence into batches
        seq_len = input_ids.size(1)
        num_batches = (seq_len + batch_size - 1) // batch_size
        input_ids_batches = [
            input_ids[:, i * batch_size : (i + 1) * batch_size]
            for i in range(num_batches)
        ]

        # Create a DataLoader for the input batches
        input_dataset = InputDataset(input_ids_batches)
        input_dataloader = DataLoader(
            input_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # Initialize an empty tensor to store the accumulated SNN output
        accumulated_snn_output = torch.zeros(
            input_ids.size(0),
            seq_len,
            self.snn_params["output_size"],
            device=device,
        )

        # Iterate over the input batches
        for batch_input_ids in input_dataloader:
            batch_input_ids = batch_input_ids.to(device)

            # Dynamically initialize the SNNLayer based on the input size
            if self.snn_layer is None:
                input_size = batch_input_ids.size(-1)
                self.snn_layer = SNNLayer(
                    input_size=input_size,
                    hidden_size=self.snn_params["hidden_size"],
                    output_size=self.snn_params["output_size"],
                    timesteps=self.snn_params["timesteps"],
                    dropout=self.snn_params.get("dropout", 0.0),
                ).to(device)

            # Apply the SNN layer
            snn_output = self.snn_layer(batch_input_ids)
            logger.debug(f"SNN output shape: {snn_output.shape}")

            # Ensure the SNN output tensor has the expected shape
            expected_snn_output_shape = (
                batch_input_ids.size(0),
                self.snn_params["timesteps"],
                batch_input_ids.size(1),
                self.snn_params["output_size"],
            )
            if snn_output.shape != expected_snn_output_shape:
                raise ValueError(
                    f"SNN output shape {snn_output.shape} "
                    f"does not match the expected shape {expected_snn_output_shape}"
                )

            # Reshape the SNN output to (batch_size, seq_len, snn_output_size)
            snn_output = snn_output[:, -1, :, :]  # Take the last timestep output

            # Accumulate the SNN output for the current batch
            start_idx = batch_input_ids[0, 0].item()
            end_idx = batch_input_ids[0, -1].item() + 1
            accumulated_snn_output[:, start_idx:end_idx, :] = snn_output

            # Initialize an empty tensor to store the accumulated GAT output
            accumulated_gat_output = torch.zeros(
                input_ids.size(0),
                seq_len,
                self.gat_params["out_features"]
                if not self.concat
                else self.gat_params["num_heads"] * self.gat_params["out_features"],
                device=device,
            )

            # Process the adjacency matrix in chunks
            if adjacency_matrix_batches is None:
                # If no adjacency matrix is provided, create a dummy identity matrix
                dummy_adj_matrix = torch.eye(seq_len, device=device)
                adjacency_matrix_chunks = [dummy_adj_matrix]
            else:
                adjacency_matrix_chunks = self.process_adjacency_matrix_in_chunks(
                    adjacency_matrix_batches, seq_len
                )

            for batch_adj_matrix in adjacency_matrix_chunks:
                # Apply the GAT layer
                gat_output = self.gat_layer(accumulated_snn_output, batch_adj_matrix)
                ...

                # Accumulate the GAT output
                accumulated_gat_output += gat_output

            # Compute the average GAT output
            avg_gat_output = accumulated_gat_output / (
                len(adjacency_matrix_chunks) or 1
            )
            logger.debug(f"Average GAT output shape: {avg_gat_output.shape}")

            # Apply the HTM layer
            htm_output = self.htm_layer(avg_gat_output).view(input_ids.size(0), -1)
            logger.debug(f"HTM output shape: {htm_output.shape}")

            # Apply the final fully connected layer
            output = self.fc_out(htm_output)
            logger.debug(f"Final output shape: {output.shape}")

            return output

    def process_adjacency_matrix_in_chunks(
        self,
        batch_adj_matrix: Union[
            torch.Tensor, torch.sparse.Tensor, Generator[torch.Tensor, None, None]
        ],
        seq_len: int,
        chunk_size: int = 1024,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Process a batch of adjacency matrices or a single adjacency matrix in chunks to avoid memory issues.

        Args:
            batch_adj_matrix (torch.Tensor, torch.sparse.Tensor, or Generator[torch.Tensor, None, None]): Batch of adjacency matrices
                with shape (batch_size, seq_len, seq_len) or a single adjacency matrix
                with shape (seq_len, seq_len), or a generator of adjacency matrix chunks.
            seq_len (int): The sequence length.
            chunk_size (int): Number of adjacency matrix elements to process
                in each chunk (default: 1024).

        Yields:
            torch.Tensor: Processed adjacency matrix chunk.
        """
        if isinstance(batch_adj_matrix, torch.sparse.Tensor):
            # If the input is a sparse tensor, process it directly without converting to dense
            batch_size = batch_adj_matrix.size(0)

            # Move the sparse tensor to the GPU
            batch_adj_matrix = batch_adj_matrix.to(self.device)

            # Process the sparse tensor in chunks
            for i in range(batch_size):
                adj_matrix = batch_adj_matrix[i]

                # Initialize output tensor for the current batch
                output = torch.zeros((seq_len, seq_len), device=self.device)

                for start_row in range(0, seq_len, chunk_size):
                    end_row = min(start_row + chunk_size, seq_len)

                    for start_col in range(0, seq_len, chunk_size):
                        end_col = min(start_col + chunk_size, seq_len)

                        # Get the current chunk of the sparse tensor
                        adj_matrix_chunk = adj_matrix[
                            start_row:end_row, start_col:end_col
                        ]

                        # Check if the chunk is empty
                        if adj_matrix_chunk.nnz() == 0:
                            continue

                        # Normalize the current chunk by row using L1 normalization
                        adj_matrix_chunk = torch.sparse.softmax(
                            adj_matrix_chunk, dim=-1
                        )

                        # Convert the sparse chunk to dense and update the corresponding part of the output tensor
                        output[
                            start_row:end_row, start_col:end_col
                        ] = adj_matrix_chunk.to_dense()

                # Yield the processed adjacency matrix for the current batch
                yield output

        elif isinstance(batch_adj_matrix, torch.Tensor):
            # If the input is a dense tensor
            if batch_adj_matrix.dim() == 2:
                # If the input is a 2D tensor, add a batch dimension
                batch_adj_matrix = batch_adj_matrix.unsqueeze(0)

            batch_size, seq_len, _ = batch_adj_matrix.shape

            # Move the dense tensor to the GPU
            batch_adj_matrix = batch_adj_matrix.to(self.device)

            # Process adjacency matrix in chunks
            for batch_idx in range(batch_size):
                batch_matrix = batch_adj_matrix[batch_idx]

                for start_row in range(0, seq_len, chunk_size):
                    end_row = min(start_row + chunk_size, seq_len)

                    for start_col in range(0, seq_len, chunk_size):
                        end_col = min(start_col + chunk_size, seq_len)

                        # Get the current chunk
                        adj_matrix_chunk = batch_matrix[
                            start_row:end_row, start_col:end_col
                        ]

                        # Normalize the current chunk by row using L1 normalization
                        adj_matrix_chunk = torch.nn.functional.normalize(
                            adj_matrix_chunk, p=1, dim=-1
                        )

                        # Update the corresponding part of the batch adjacency matrix with the processed chunk
                        batch_matrix[
                            start_row:end_row, start_col:end_col
                        ] = adj_matrix_chunk

                # Yield the processed batch adjacency matrix
                yield batch_matrix

        elif isinstance(batch_adj_matrix, Generator):
            # If the input is a generator, iterate over it and yield each tensor
            for adj_matrix_chunk in batch_adj_matrix:
                yield adj_matrix_chunk.to(self.device)

        else:
            raise TypeError(
                "Input must be a torch.Tensor, torch.sparse.Tensor, or a Generator of torch.Tensor"
            )

    def generate(
            self,
            input_conversation: list,
            adjacency_matrix: torch.Tensor,
            node_indices: torch.Tensor,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            mirostat_eta: float = 0.1,
            mirostat_tau: float = 5.0,
            **_,
    ) -> list:
        """
        Generate responses based on the input conversation.

        Args:
            input_conversation (list): The input conversation as a list of strings or tensors.
            adjacency_matrix (torch.Tensor): The adjacency matrix tensor.
            node_indices (torch.Tensor): The node indices tensor.
            max_length (int): The maximum length of the generated response.
            num_return_sequences (int): The number of responses to generate.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top-k tokens to consider for filtering.
            top_p (float): The cumulative probability threshold for filtering.
            mirostat_eta (float): The learning rate for Mirostat.
            mirostat_tau (float): The target entropy for Mirostat.

        Returns:
            list: The generated conversations as a list of strings.
        """
        generated_conversations = []

        # Convert the input conversation to a list of strings
        string_conversation = [
            str(turn.item()) if isinstance(turn, torch.Tensor) else str(turn)
            for turn in input_conversation
        ]

        # Check if the input conversation is empty
        if not string_conversation:
            logger.warning("Input conversation is empty. Skipping generation.")
            return generated_conversations

        for _ in range(num_return_sequences):
            generated_conversation = []

            # Concatenate the input conversation to start the generation
            input_text = " ".join(string_conversation)
            generated_conversation.append(input_text)

            # Initialize Mirostat parameters
            mirostat_mu = 2.0 * mirostat_tau
            mirostat_s = 1.0

            # Tokenize the input conversation
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.device)

            response = []
            while len(response) < max_length:
                # Forward pass
                if adjacency_matrix is None or node_indices is None:
                    # If adjacency_matrix or node_indices are not provided, create dummy values
                    dummy_adj_matrix = torch.eye(
                        input_ids.size(-1), dtype=torch.bool
                    ).to(self.device)
                    output = self.forward(input_ids, dummy_adj_matrix)
                else:
                    output = self.forward(input_ids, adjacency_matrix)

                # Sample from the model's output distribution
                output_logits = output[:, -1, :] / temperature
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

                # Check if the generated token is the end-of-sequence token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                response.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.tensor([1], device=self.device, dtype=torch.long),
                    ],
                    dim=-1,
                )

            # Add the generated response to the conversation
            generated_response = self.tokenizer.decode(response)
            generated_conversation.append(generated_response)
            generated_conversations.append(generated_conversation)

        return generated_conversations

class InputDataset(Dataset):
    def __init__(self, input_ids_batches):
        """
        Initialize the InputDataset.

        Args:
            input_ids_batches (List[torch.Tensor]): A list of input ID tensors, where each tensor represents a batch of input sequences.
        """
        self.input_ids_batches = input_ids_batches

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of batches in the dataset.
        """
        return len(self.input_ids_batches)

    def __getitem__(self, idx):
        """
        Get a batch of input sequences from the dataset.

        Args:
            idx (int): The index of the batch to retrieve.

        Returns:
            torch.Tensor: A tensor containing a batch of input sequences.
        """
        return self.input_ids_batches[idx]
