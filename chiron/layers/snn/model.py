# chiron/layers/snn/model.py

from typing import Dict, Union, Optional

import plotly.graph_objects as go
import scipy
import torch
import torch.nn as nn
from loguru import logger
from plotly.subplots import make_subplots

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


class SNNModel(nn.Module):
    def __init__(
        self,
        sp_params: Dict,
        gat_params: Dict,
        htm_params: Dict,
        snn_params: Dict,
        device: torch.device,
    ):
        """
        Initialize the SNNModel.

        Args:
            sp_params (Dict): Parameters for the Sparse Distributed Representation (SDR) generation.
            gat_params (Dict): Parameters for the Graph Attention Network (GAT) layer.
            htm_params (Dict): Parameters for the Hierarchical Temporal Memory (HTM) layer.
            snn_params (Dict): Parameters for the Spiking Neural Network (SNN) layer.
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

        self.visualization_data = None

        # Initialize the SNN layer
        self.snn_layer = SNNLayer(
            input_size=sp_params["sdr_dimensions"],
            hidden_size=snn_params["hidden_size"],
            output_size=snn_params["output_size"],
            timesteps=snn_params["timesteps"],
            dropout=snn_params.get("dropout", 0.0),
        ).to(device)

        # Initialize the GAT layer
        self.gat_layer = GraphAttentionLayer(
            in_features=snn_params["output_size"],
            out_features=gat_params["out_features"],
            num_heads=gat_params["num_heads"],
            alpha=gat_params["alpha"],
            concat=self.concat,
            fallback_mode=gat_params.get("fallback_mode", "dense"),
        ).to(device)

        # Initialize the HTM layer
        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        # Initialize the fully connected layer with the output size of the SDR layer
        self.fc_out = nn.Linear(sp_params["sdr_dimensions"], self.output_size).to(
            device
        )

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, gat_params={gat_params},"
            f"htm_params={htm_params}, output_size={self.output_size}"
        )

    def visualize(self, input_ids: Optional[torch.Tensor] = None) -> None:
        """
        Visualize the neural network architecture, data flow, and model parameters using Plotly.
        """
        if input_ids is None:
            if self.visualization_data is None:
                raise ValueError(
                    "No input data available for visualization. Please run the forward pass first."
                )

        input_ids = self.visualization_data["input_ids"]
        attention_mask = self.visualization_data["attention_mask"]
        snn_output = self.visualization_data["snn_output"]
        gat_output = self.visualization_data["gat_output"]
        htm_output = self.visualization_data["htm_output"]
        final_output = self.visualization_data["final_output"]
        sdr_embeddings = self.visualization_data["sdr_embeddings"]
        adjacency_matrix = self.visualization_data["adjacency_matrix"]

        # Visualize the input data
        self.visualize_input_data(input_ids, attention_mask)

        # Visualize the SDR embeddings
        self.visualize_sdr_embeddings(sdr_embeddings)

        # Visualize the adjacency matrix
        self.visualize_adjacency_matrix(adjacency_matrix)

        # Visualize the model architecture and data flow
        self.visualize_model_architecture(
            input_ids, snn_output, gat_output, htm_output, final_output
        )

    def visualize_input_data(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        """
        Visualize the input data, including the token IDs and attention mask.
        """
        # batch_size, seq_len = input_ids.shape
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Token IDs", "Attention Mask"]
        )

        # Visualize the token IDs
        token_ids_trace = go.Heatmap(
            z=input_ids.cpu().numpy(),
            colorscale="Viridis",
            zmin=0,
            zmax=self.tokenizer.vocab_size,
        )
        fig.add_trace(token_ids_trace, row=1, col=1)

        # Visualize the attention mask
        attention_mask_trace = go.Heatmap(
            z=attention_mask.cpu().numpy(),
            colorscale="Greys",
            zmin=0,
            zmax=1,
        )
        fig.add_trace(attention_mask_trace, row=1, col=2)

        fig.update_layout(
            title="Input Data",
            height=400,
            width=800,
            showlegend=False,
        )

        fig.show()
        fig.write_image(".visualizations/input_data.png")

    @classmethod
    def visualize_sdr_embeddings(cls, sdr_embeddings: torch.Tensor) -> None:
        """
        Visualize the SDR embeddings.
        """
        num_embeddings, sdr_dim = sdr_embeddings.shape
        fig = go.Figure()

        for i in range(
            min(num_embeddings, 1000)
        ):  # Limit the number of embeddings to visualize
            embedding = sdr_embeddings[i].cpu().numpy()
            x = [f"Dim {j}" for j in range(sdr_dim)]
            y = embedding
            fig.add_trace(go.Bar(x=x, y=y, name=f"Embedding {i}"))

        fig.update_layout(
            title="SDR Embeddings",
            xaxis_title="Dimensions",
            yaxis_title="Values",
            barmode="stack",
            height=600,
            width=800,
        )

        fig.show()
        fig.write_image(".visualizations/sdr_embeddings.png")


    @classmethod
    def visualize_adjacency_matrix(
        cls, adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor]
    ) -> None:
        """
        Visualize the adjacency matrix.
        """
        if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
            adjacency_matrix = adjacency_matrix.toarray()

        fig = go.Figure(
            data=go.Heatmap(
                z=adjacency_matrix.cpu().numpy(),
                colorscale="Viridis",
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title="Adjacency Matrix",
            xaxis_title="Nodes",
            yaxis_title="Nodes",
            height=600,
            width=800,
        )

        fig.show()
        fig.write_image(".visualizations/adj_matrix")

    def visualize_model_architecture(
        self,
        input_ids: torch.Tensor,
        snn_output: torch.Tensor,
        gat_output: torch.Tensor,
        htm_output: torch.Tensor,
        final_output: torch.Tensor,
    ) -> None:
        """
        Visualize the model architecture and data flow through the layers.
        """
        batch_size, seq_len = input_ids.shape
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Input Token IDs",
                "SNN Output",
                "GAT Output",
                "HTM Output",
                "Final Output",
                "Output Distributions",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        # Visualize the input token IDs
        input_ids_trace = go.Heatmap(
            z=input_ids.cpu().numpy(),
            colorscale="Viridis",
            zmin=0,
            zmax=self.tokenizer.vocab_size,
        )
        fig.add_trace(input_ids_trace, row=1, col=1)

        # Visualize the SNN output
        snn_output_trace = go.Heatmap(
            z=snn_output.cpu()
            .numpy()
            .mean(dim=2),  # Take the mean across the timestep dimension
            colorscale="Plasma",
        )
        fig.add_trace(snn_output_trace, row=1, col=2)

        # Visualize the GAT output
        gat_output_trace = go.Heatmap(
            z=gat_output.cpu().numpy(),
            colorscale="Viridis",
        )
        fig.add_trace(gat_output_trace, row=1, col=3)

        # Visualize the HTM output
        htm_output_trace = go.Heatmap(
            z=htm_output.cpu().numpy(),
            colorscale="Inferno",
        )
        fig.add_trace(htm_output_trace, row=2, col=1)

        # Visualize the final output
        final_output_trace = go.Heatmap(
            z=final_output.cpu().numpy(),
            colorscale="Viridis",
        )
        fig.add_trace(final_output_trace, row=2, col=2)

        # Visualize the output distributions
        output_distributions = []
        for i in range(batch_size):
            output_distribution = torch.softmax(final_output[i], dim=0).cpu().numpy()
            output_distributions.append(output_distribution)

        output_distributions_trace = go.Violin(
            y=output_distributions,
            box_visible=True,
            meanline_visible=True,
        )
        fig.add_trace(output_distributions_trace, row=2, col=3)

        fig.update_layout(
            title="Model Architecture and Data Flow",
            height=800,
            width=1200,
            showlegend=False,
        )

        fig.show()
        fig.write_image(".visualizations/model_architecture")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor],
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the SNNModel.

        Args:
            input_ids (torch.Tensor): The input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
            adjacency_matrix (Union[scipy.sparse.csr_matrix, torch.Tensor]): The adjacency matrix as a SciPy sparse matrix or PyTorch tensor.
            node_indices (torch.Tensor): The node indices of shape (batch_size,).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        # Reshape the input tensor to (batch_size, seq_len, input_size)
        input_size = self.snn_params["input_size"]
        input_ids = input_ids.unsqueeze(-1).expand(*input_ids.shape, input_size)

        # Apply the SNN layer
        snn_output = self.snn_layer(input_ids)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Reshape the SNN output to (batch_size, seq_len, snn_output_size)
        batch_size, seq_len, timesteps, snn_output_size = snn_output.shape
        snn_output = snn_output.view(batch_size, seq_len, timesteps * snn_output_size)

        # Apply the GAT layer
        seq_len = snn_output.size(1)
        adj_matrix_tensor = adjacency_matrix[
            :seq_len, :seq_len
        ]  # Slice the adjacency matrix to match the sequence length
        gat_output = self.gat_layer(snn_output, adj_matrix_tensor)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Reshape the attention mask to match the expected shape (batch_size, seq_len)
        expected_attention_mask_shape = (batch_size, seq_len)
        if attention_mask.shape != expected_attention_mask_shape:
            logger.debug(
                f"Attention mask shape {attention_mask.shape} "
                f"does not match the expected shape {expected_attention_mask_shape}. "
                f"Reshaping the attention mask."
            )
            attention_mask = attention_mask.view(expected_attention_mask_shape)
        logger.debug(f"Attention mask shape: {attention_mask.shape}")

        # Expand the attention mask to match the shape of gat_output
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            batch_size, seq_len, gat_output.size(-1)
        )
        gat_output = gat_output * attention_mask_expanded

        # Apply the HTM layer
        htm_output = self.htm_layer(gat_output)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Apply node indices to the HTM output
        node_indices_expanded = node_indices.unsqueeze(-1).expand(
            batch_size, 1, htm_output.size(-1)
        )
        htm_output_selected = torch.gather(htm_output, 1, node_indices_expanded)
        htm_output_selected = htm_output_selected.squeeze(1)

        # Apply the final fully connected layer
        output = self.fc_out(htm_output_selected)
        logger.debug(f"Final output shape: {output.shape}")
        # Store the intermediate outputs and data details for visualization

        self.visualization_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "snn_output": snn_output,
            "gat_output": gat_output,
            "htm_output": htm_output,
            "final_output": output,
            "sdr_embeddings": self.sp_params["sdr_embeddings"],
            "adjacency_matrix": adjacency_matrix,
        }

        return output


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
