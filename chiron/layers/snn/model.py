# chiron/layers/snn/model.py

# chiron/layers/snn/model.py

from typing import List

import torch
import torch.nn as nn
from loguru import logger

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer
from chiron.utils.tokenization import EmbeddingTokenizer


class SNNModel(nn.Module):
    def __init__(
            self,
            sp_params: dict,
            gat_params: dict,
            htm_params: dict,
            snn_params: dict,
            device: torch.device,
            tokenizer: EmbeddingTokenizer,
    ):
        """
        Initialize the SNNModel.

        Args:
            sp_params (dict): Parameters for the Sparse Distributed Representation (SDR) generation.
            gat_params (dict): Parameters for the Graph Attention Network (GAT) layer.
            htm_params (dict): Parameters for the Hierarchical Temporal Memory (HTM) layer.
            snn_params (dict): Parameters for the Spiking Neural Network (SNN) layer.
            device (torch.device): The device to use for model operations.
            tokenizer (EmbeddingTokenizer): The tokenizer for encoding and decoding embeddings.
        """
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.snn_params = snn_params
        self.device = device
        self.tokenizer = tokenizer
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

        # Initialize the fully connected layer with the output size of the SDR layer
        self.fc_out = nn.Linear(sp_params["sdr_dimensions"], self.output_size).to(device)

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
        batch_size, seq_len = input_ids.shape

        # Decode the input IDs to obtain the binarized embeddings
        input_embeddings = self.tokenizer.decode(input_ids.cpu().numpy())
        input_embeddings = torch.tensor(input_embeddings, dtype=torch.float32).to(self.device)

        # Reshape the input tensor to (batch_size, seq_len, input_size)
        input_size = self.snn_params["input_size"]
        input_embeddings = input_embeddings.unsqueeze(-1).expand(*input_embeddings.shape, input_size)

        # Apply the SNN layer
        snn_output = self.snn_layer(input_embeddings)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Reshape the SNN output to (batch_size, seq_len, snn_output_size)
        batch_size, seq_len, timesteps, snn_output_size = snn_output.shape
        snn_output = snn_output.view(batch_size, seq_len, timesteps * snn_output_size)

        # Apply the GAT layer
        seq_len = snn_output.size(1)
        adj_matrix_tensor = adjacency_matrix[:seq_len,
                            :seq_len]  # Slice the adjacency matrix to match the sequence length
        gat_output = self.gat_layer(snn_output, adj_matrix_tensor)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Apply attention mask to the GAT output
        attention_mask_expanded = attention_mask[:, :seq_len].unsqueeze(-1).expand_as(gat_output)
        gat_output = gat_output * attention_mask_expanded

        # Apply the HTM layer
        htm_output = self.htm_layer(gat_output)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Apply node indices to the HTM output
        node_indices_expanded = node_indices[:, :seq_len].unsqueeze(-1).expand(batch_size, seq_len, htm_output.size(-1))
        htm_output_selected = torch.gather(htm_output, 1, node_indices_expanded)
        htm_output_selected = htm_output_selected.squeeze(1)

        # Apply the final fully connected layer
        output = self.fc_out(htm_output_selected)
        logger.debug(f"Final output shape: {output.shape}")

        return output

    def generate(
            self,
            input_conversation: List[str],
            adjacency_matrix: torch.Tensor,
            node_indices: torch.Tensor,
            attention_mask: torch.Tensor,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            mirostat_eta: float = 0.1,
            mirostat_tau: float = 5.0,
            **_,
    ) -> List[str]:
        """
        Generate responses based on the input conversation.

        Args:
            input_conversation (List[str]): The input conversation as a list of strings.
            adjacency_matrix (torch.Tensor): The adjacency matrix tensor.
            node_indices (torch.Tensor): The node indices tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            max_length (int): The maximum length of the generated response.
            num_return_sequences (int): The number of responses to generate.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top-k tokens to consider for filtering.
            top_p (float): The cumulative probability threshold for filtering.
            mirostat_eta (float): The learning rate for Mirostat.
            mirostat_tau (float): The target entropy for Mirostat.

        Returns:
            List[str]: The generated conversations as a list of strings.
        """
        generated_conversations = []

        # Encode the input conversation to obtain the input IDs
        input_ids = self.tokenizer.encode(input_conversation)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)

        batch_size, seq_len = input_ids.shape

        # Initialize Mirostat parameters
        mirostat_mu = 2.0 * mirostat_tau
        mirostat_s = 1.0

        for _ in range(num_return_sequences):
            response = []
            curr_input_ids = input_ids.clone()
            curr_attention_mask = attention_mask.clone()

            while len(response) < max_length:
                curr_adjacency_matrix = adjacency_matrix[:seq_len, :seq_len]  # Slice adjacency matrix
                curr_node_indices = node_indices[:, :seq_len]  # Slice node indices

                # Forward pass
                output = self.forward(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_adjacency_matrix,
                    curr_node_indices,
                )

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
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
                curr_attention_mask = torch.cat(
                    [curr_attention_mask, torch.tensor([[1]], device=curr_attention_mask.device)], dim=-1
                )

                seq_len += 1  # Increment sequence length

            # Decode the generated response
            generated_response = self.tokenizer.decode(response)
            generated_conversations.append(generated_response)

        return generated_conversations

    @staticmethod
    def _top_k_top_p_filtering(
            logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9
    ) -> torch.Tensor:
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
        sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_indices = sorted_indices + sorted_indices_to_remove.long()

        # Create a mask to filter out the indices that exceed top_p
        mask = sorted_indices != sorted_indices
        top_p_logits = top_k_logits.masked_fill(mask, float("-inf"))

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
            torch.Tensor: Output tensor of shape (batch_size, seq_len, timesteps, output_size).
        """
        batch_size, seq_len, input_size = x.shape

        # Reshape the input tensor to (batch_size * seq_len, input_size)
        x = x.view(batch_size * seq_len, input_size)

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
        output = torch.stack(spikes, dim=2)

        return output
