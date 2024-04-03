# chiron/layers/snn/model.py
from contextlib import contextmanager
from typing import Optional, Generator, Union

import torch
import torch.nn as nn
from loguru import logger
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
    ):
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.output_size = sp_params["sdr_dimensions"]
        self.snn_params = snn_params

        self.snn_layer = None
        self.gat_layer = GraphAttentionLayer(
            in_features=self.snn_params["output_size"],
            out_features=self.gat_params["out_features"],
            num_heads=self.gat_params["num_heads"],
            alpha=self.gat_params["alpha"],
            concat=self.gat_params["concat"],
        ).to(device)

        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        self.fc_out = nn.Linear(self.output_size, self.output_size).to(device)

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
        logger.debug(f"Input tensor shape: {input_ids.shape}")

        batch_size, seq_len = input_ids.size()

        # Move the input tensor to the specified device
        input_ids = input_ids.to(self.device)

        # Reshape the input tensor to (batch_size, seq_len, input_size)
        input_ids = input_ids.view(batch_size, seq_len, -1)

        # Dynamically initialize the SNNLayer based on the input size
        if self.snn_layer is None:
            input_size = input_ids.size(-1)
            self.snn_layer = SNNLayer(
                input_size=input_size,
                hidden_size=self.snn_params["hidden_size"],
                output_size=self.snn_params["output_size"],
                timesteps=self.snn_params["timesteps"],
                dropout=self.snn_params.get("dropout", 0.0),
            ).to(self.device)

        # Apply the SNN layer
        snn_output = self.snn_layer(input_ids)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Ensure the SNN output tensor has the expected shape
        expected_snn_output_shape = (
            batch_size,
            self.snn_params["timesteps"],
            seq_len,
            self.snn_params["output_size"],
        )
        if snn_output.shape != expected_snn_output_shape:
            raise ValueError(
                f"SNN output shape {snn_output.shape} "
                f"does not match the expected shape {expected_snn_output_shape}"
            )

        # Reshape the SNN output to (batch_size, seq_len, snn_output_size)
        snn_output = snn_output[:, -1, :, :]  # Take the last timestep output

        # Initialize an empty tensor to store the accumulated GAT output
        accumulated_gat_output = torch.zeros(
            batch_size, seq_len, self.gat_params["out_features"], device=self.device
        )

        # Process each batch adjacency matrix tensor
        if adjacency_matrix_batches is None:
            # If no adjacency matrix batches are provided, create a dummy identity matrix
            dummy_adj_matrix = torch.eye(seq_len, device=self.device)
            gat_output = self.gat_layer(snn_output, dummy_adj_matrix)
            accumulated_gat_output += gat_output
        else:
            for batch_adj_matrix in adjacency_matrix_batches:
                # Ensure the batch adjacency matrix has the correct number of dimensions
                if batch_adj_matrix.ndim != 2:
                    raise ValueError(
                        f"Batch adjacency matrix must be a 2D tensor, but got {batch_adj_matrix.ndim}D tensor"
                    )

                # Ensure the batch adjacency matrix has the correct shape
                expected_shape = (seq_len, seq_len)
                if batch_adj_matrix.shape != expected_shape:
                    # If the batch adjacency matrix has a different shape, process it in chunks
                    batch_adj_matrix = self.process_adjacency_matrix_in_chunks(
                        batch_adj_matrix, seq_len
                    )

                # Apply the GAT layer
                gat_output = self.gat_layer(snn_output, batch_adj_matrix)
                logger.debug(f"GAT output shape: {gat_output.shape}")

                # Ensure the GAT output tensor has the expected shape
                expected_gat_output_shape = (
                    batch_size,
                    seq_len,
                    self.gat_params["out_features"],
                )
                if gat_output.shape != expected_gat_output_shape:
                    raise ValueError(
                        f"GAT output shape {gat_output.shape} does not match the expected shape {expected_gat_output_shape}"
                    )

                # Accumulate the GAT output
                accumulated_gat_output += gat_output

        # Compute the average GAT output
        avg_gat_output = accumulated_gat_output / (len(adjacency_matrix_batches) or 1)
        logger.debug(f"Average GAT output shape: {avg_gat_output.shape}")

        # Apply the HTM layer
        htm_output = self.htm_layer(avg_gat_output).view(batch_size, -1)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Apply the final fully connected layer
        output = self.fc_out(htm_output)
        logger.debug(f"Final output shape: {output.shape}")

        return output

    def process_adjacency_matrix_in_chunks(
            self,
            batch_adj_matrix: Union[torch.Tensor, torch.sparse.Tensor],
            seq_len: int,
            chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Process a batch of adjacency matrices or a single adjacency matrix in chunks to avoid memory issues.

        Args:
            batch_adj_matrix (torch.Tensor or torch.sparse.Tensor): Batch of adjacency matrices
                with shape (batch_size, seq_len, seq_len) or a single adjacency matrix
                with shape (seq_len, seq_len).
            seq_len (int): The sequence length.
            chunk_size (int): Number of adjacency matrix elements to process
                in each chunk (default: 1024).

        Returns:
            torch.Tensor: Processed adjacency matrices or a single adjacency matrix
                with the same shape as the input.
        """
        if isinstance(batch_adj_matrix, torch.sparse.Tensor):
            # If the input is a sparse tensor, process it directly without converting to dense
            batch_size = batch_adj_matrix.size(0)
            output = torch.zeros((batch_size, seq_len, seq_len), device=self.device)

            # Process the sparse tensor in chunks
            for i in range(batch_size):
                adj_matrix = batch_adj_matrix[i]

                for start_row in range(0, seq_len, chunk_size):
                    end_row = min(start_row + chunk_size, seq_len)

                    for start_col in range(0, seq_len, chunk_size):
                        end_col = min(start_col + chunk_size, seq_len)

                        # Get the current chunk of the sparse tensor
                        adj_matrix_chunk = adj_matrix[start_row:end_row, start_col:end_col]

                        # Normalize the current chunk by row using L1 normalization
                        adj_matrix_chunk = torch.sparse.softmax(adj_matrix_chunk, dim=-1)

                        # Convert the sparse chunk to dense and update the corresponding part of the output tensor
                        output[i, start_row:end_row, start_col:end_col] = adj_matrix_chunk.to_dense()

        else:
            # If the input is a dense tensor
            if batch_adj_matrix.dim() == 2:
                # If the input is a 2D tensor, add a batch dimension
                batch_adj_matrix = batch_adj_matrix.unsqueeze(0)

            batch_size, seq_len, _ = batch_adj_matrix.shape

            # Initialize output tensor
            output = torch.empty((batch_size, seq_len, seq_len), device=self.device)

            # Process adjacency matrix in chunks
            for start_row in range(0, seq_len, chunk_size):
                end_row = min(start_row + chunk_size, seq_len)

                for start_col in range(0, seq_len, chunk_size):
                    end_col = min(start_col + chunk_size, seq_len)

                    # Get the current chunk
                    adj_matrix_chunk = batch_adj_matrix[:, start_row:end_row, start_col:end_col]

                    # Normalize the current chunk by row using L1 normalization
                    adj_matrix_chunk = torch.nn.functional.normalize(
                        adj_matrix_chunk, p=1, dim=-1
                    )

                    # Update the corresponding part of the output tensor with the processed chunk
                    output[:, start_row:end_row, start_col:end_col] = adj_matrix_chunk

        # Remove the batch dimension if the input was a single adjacency matrix
        if output.size(0) == 1:
            output = output.squeeze(0)

        return output


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

            # Generate the model's response
            model_input = self._prepare_input(generated_conversation)
            input_ids = model_input["input_ids"].to(self.device)
            attention_mask = model_input["attention_mask"].to(self.device)

            response = []
            while len(response) < max_length:
                # Forward pass
                if adjacency_matrix is None or node_indices is None:
                    # If adjacency_matrix or node_indices are not provided, create dummy values
                    dummy_adj_matrix = torch.eye(
                        input_ids.size(-1), dtype=torch.bool
                    ).to(self.device)

                    # TODO: Resolve the model not using attention mask and node indices properly in some cases
                    # dummy_node_indices = torch.arange(
                    #     input_ids.size(-1), device=self.device
                    # )
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

    def _prepare_input(self, conversation: list) -> dict:
        """
        Prepare the input for the model based on the conversation structure.

        Args:
            conversation (list): The conversation as a list of strings.

        Returns:
            dict: The prepared input for the model.
        """
        input_ids = []

        for turn in conversation:
            if isinstance(turn, str):
                encoded_turn = self.tokenizer.encode(turn, add_special_tokens=False)
                if encoded_turn:
                    input_ids.extend(encoded_turn + [self.tokenizer.eos_token_id])

        # Remove None values from input_ids
        input_ids = [token for token in input_ids if token is not None]

        if input_ids:
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            logger.warning(
                "No valid turns found in the conversation. Returning empty input."
            )
            empty_tensor = torch.tensor([], dtype=torch.long).unsqueeze(0)
            return {"input_ids": empty_tensor, "attention_mask": empty_tensor}

    @staticmethod
    def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        """
        Filter the logits using top-k and/or top-p filtering.

        Args:
            logits (torch.Tensor): The logits tensor.
            top_k (int, optional): The number of top-k tokens to keep. Default is 0.
            top_p (float, optional): The cumulative probability threshold for top-p filtering. Default is 1.0.
            filter_value (float, optional): The value to replace the filtered logits with. Default is -infinity.

        Returns:
            torch.Tensor: The filtered logits tensor.
        """
        # Filter the logits using top-k and/or top-p filtering
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
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

            # Convert from sorted indices to original indices
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits

    @contextmanager
    def torch_no_grad_and_inference_mode(self, device: torch.device):
        """
        Context manager to temporarily set PyTorch to no_grad and inference mode.

        Args:
            device (torch.device): The device to use for inference.

        Yields:
            None
        """
        with torch.no_grad(), torch.inference_mode():
            yield
