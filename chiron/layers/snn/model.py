# chiron/layers/snn/model.py

from typing import List

import torch
import torch.nn as nn
from loguru import logger

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


# chiron/layers/snn/model.py

from typing import List

import torch
import torch.nn as nn
from loguru import logger

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


class SNNModel(nn.Module):
    def __init__(
            self,
            sp_params: dict,
            gat_params: dict,
            htm_params: dict,
            snn_params: dict,
            device: torch.device,
    ):
        """
        Initialize the SNNModel.

        Args:
            sp_params (dict): Parameters for the Sparse Distributed Representation (SDR) generation.
            gat_params (dict): Parameters for the Graph Attention Network (GAT) layer.
            htm_params (dict): Parameters for the Hierarchical Temporal Memory (HTM) layer.
            snn_params (dict): Parameters for the Spiking Neural Network (SNN) layer.
            device (torch.device): The device to use for model operations.
        """
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.snn_params = snn_params
        self.device = device
        self.output_size = sp_params["sdr_dimensions"]
        self.concat = gat_params["concat"]

        self.snn_layer = SNNLayer(
            input_size=sp_params["sdr_dimensions"],
            hidden_size=snn_params["hidden_size"],
            output_size=snn_params["output_size"],
            timesteps=snn_params["timesteps"],
            dropout=snn_params.get("dropout", 0.0),
        ).to(device)

        self.gat_layer = GraphAttentionLayer(
            in_features=snn_params["output_size"],
            out_features=gat_params["out_features"],
            num_heads=gat_params["num_heads"],
            alpha=gat_params["alpha"],
            concat=self.concat,
            fallback_mode=gat_params.get("fallback_mode", "dense"),
        ).to(device)

        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        if self.concat:
            self.fc_out = nn.Linear(
                gat_params["num_heads"] * gat_params["out_features"],
                self.output_size,
            ).to(device)
        else:
            self.fc_out = nn.Linear(
                gat_params["out_features"], self.output_size
            ).to(device)

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, gat_params={gat_params},"
            f"htm_params={htm_params}, output_size={self.output_size}"
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            adjacency_matrix: torch.Tensor,
            node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the SNN model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len).
            adjacency_matrix (torch.Tensor): Adjacency matrix tensor of shape (seq_len, seq_len).
            node_indices (torch.Tensor): Node indices tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).
        """
        device = self.device
        batch_size, seq_len = input_ids.shape

        # Reshape the input tensor to (batch_size, seq_len, input_size)
        input_size = self.snn_params["input_size"]
        input_ids = input_ids.unsqueeze(-1).expand(*input_ids.shape, input_size)

        # Apply the SNN layer
        snn_output = self.snn_layer(input_ids)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Reshape the SNN output to (batch_size, seq_len, timesteps * snn_output_size)
        _, timesteps, _, snn_output_size = snn_output.shape
        snn_output = snn_output.view(batch_size, seq_len, timesteps * snn_output_size)

        # Apply attention mask to the SNN output
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(batch_size, seq_len, timesteps * snn_output_size)
        snn_output = snn_output * attention_mask_expanded

        # Apply the GAT layer
        gat_output = self.gat_layer(snn_output, adjacency_matrix)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Apply the HTM layer
        htm_output = self.htm_layer(gat_output)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Apply node indices to the HTM output
        node_indices_expanded = node_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, htm_output.size(-1))
        htm_output_selected = torch.gather(htm_output, 1, node_indices_expanded)
        htm_output_selected = htm_output_selected.squeeze(1)

        # Apply the final fully connected layer
        output = self.fc_out(htm_output_selected)
        logger.debug(f"Final output shape: {output.shape}")

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

            # Tokenize the input conversation
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.device)

            response = []
            while len(response) < max_length:
                # Forward pass
                if adjacency_matrix is None or node_indices is None:
                    # If adjacency_matrix or node_indices are not provided, create dummy values
                    dummy_adj_matrix = torch.eye(input_ids.size(-1), dtype=torch.bool).to(self.device)
                    dummy_node_indices = torch.zeros(1, dtype=torch.long).to(self.device)
                    output = self.forward(input_ids, attention_mask, [dummy_adj_matrix], dummy_node_indices)
                else:
                    output = self.forward(input_ids, attention_mask, [adjacency_matrix], node_indices)

                # Sample from the model's output distribution
                output_logits = output[:, -1, :] / temperature
                filtered_logits = self._top_k_top_p_filtering(output_logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Update Mirostat parameters
                mirostat_s = mirostat_eta * (output_logits.max().item() - mirostat_tau) + (
                        1 - mirostat_eta) * mirostat_s
                mirostat_mu = mirostat_mu * torch.exp(mirostat_s)
                temperature = max(0.1, temperature * (mirostat_tau / mirostat_mu))

                # Check if the generated token is the end-of-sequence token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                response.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor([1], device=self.device, dtype=torch.long)],
                                           dim=-1)

            # Add the generated response to the conversation
            generated_response = self.tokenizer.decode(response)
            generated_conversation.append(generated_response)
            generated_conversations.append(generated_conversation)

        return generated_conversations

    @classmethod
    def _top_k_top_p_filtering(cls, logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """
        Apply top-k and top-p filtering to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.
            top_k (int): The number of top-k tokens to consider for filtering.
            top_p (float): The cumulative probability threshold for filtering.

        Returns:
            torch.Tensor: The filtered logits tensor.
        """
        # Top-k filtering
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Top-p filtering
        top_p_logits = top_k_logits.clone()
        top_p_logits.fill_(-float("inf"))
        sorted_indices = top_k_indices.argsort(dim=-1, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(top_k_logits, dim=-1)[0, sorted_indices[0]], dim=-1)
        top_p_indices = torch.where(cumulative_probs <= top_p)[0]
        if top_p_indices.size(0) > 0:
            top_p_indices = sorted_indices[0, : top_p_indices[-1] + 1]
            top_p_logits[0, top_p_indices] = top_k_logits[0, top_p_indices]

        return top_p_logits


class SNNLayer(nn.Module):
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
            torch.Tensor: Output tensor of shape (batch_size, timesteps, seq_len, output_size).
        """
        batch_size, seq_len, input_size = x.shape

        # Reshape the input tensor to (batch_size * seq_len, input_size)
        x = x.view(batch_size * seq_len, input_size)

        # Ensure the input tensor has the same number of features as the linear layer expects
        assert x.size(
            1) == self.fc1.in_features, f"Expected input tensor to have {self.fc1.in_features} features, but got {x.size(1)}"

        # Convert the input tensor to the same dtype as the linear layer weights
        x = x.to(dtype=self.fc1.weight.dtype)

        mem1 = torch.zeros(batch_size * seq_len, self.hidden_size, device=x.device)
        mem2 = torch.zeros(batch_size * seq_len, self.output_size, device=x.device)

        spikes = []
        for _ in range(self.timesteps):
            mem1 = mem1 + self.fc1(x)
            spike1 = (mem1 > 0.5).float()  # Apply spiking behavior
            mem1 = mem1 * (mem1 <= 0.5)  # Reset membrane potential

            spike1 = self.dropout_layer(spike1)
            mem2 = mem2 + self.fc2(spike1)
            spike2 = (mem2 > 0.5).float()  # Apply spiking behavior
            mem2 = mem2 * (mem2 <= 0.5)  # Reset membrane potential

            spikes.append(spike2.view(batch_size, seq_len, self.output_size))

        # Stack the spikes along the timestep dimension
        output = torch.stack(spikes, dim=1)

        return output