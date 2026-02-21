# chiron/layers/snn/model.py

from typing import Dict, Union, Optional

import plotly.graph_objects as go
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from plotly.subplots import make_subplots

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


# ---------------------------------------------------------------------------
# Surrogate gradient: SuperSpike (Zenke & Ganguli, 2018)
# ---------------------------------------------------------------------------

class SuperSpike(torch.autograd.Function):
    """Custom autograd function implementing the SuperSpike surrogate gradient.

    Forward pass: Heaviside step function  spike = (mem > V_th).
    Backward pass: surrogate gradient  1 / (1 + k * |mem - V_th|)^2.
    """

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: torch.Tensor, k: float = 25.0):
        ctx.save_for_backward(membrane_potential, threshold)
        ctx.k = k
        return (membrane_potential > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane_potential, threshold = ctx.saved_tensors
        k = ctx.k
        # Surrogate gradient: 1 / (1 + k * |v - V_th|)^2
        v_shifted = membrane_potential - threshold
        surrogate = 1.0 / (1.0 + k * torch.abs(v_shifted)) ** 2
        grad_mem = grad_output * surrogate
        # Gradient w.r.t. threshold is the negative of the gradient w.r.t. membrane
        grad_threshold = -grad_mem.sum()
        return grad_mem, grad_threshold, None


superspike = SuperSpike.apply


# ---------------------------------------------------------------------------
# Parametric Leaky Integrate-and-Fire (PLIF) neuron layer with ALIF
# ---------------------------------------------------------------------------

class PLIFNeuronLayer(nn.Module):
    """Parametric Leaky Integrate-and-Fire neuron layer with adaptive threshold.

    Features:
        - Learnable membrane time constant (tau_mem) via sigmoid parameterisation.
        - Learnable synaptic time constant (tau_syn).
        - Learnable base firing threshold (V_th).
        - Adaptive threshold (ALIF): threshold increases after each spike and
          decays back to baseline.
        - Soft reset: membrane potential is reduced by threshold rather than
          being zeroed.
        - Optional recurrent connections for richer temporal dynamics.
        - SuperSpike surrogate gradient for back-propagation through spikes.
    """

    def __init__(
        self,
        size: int,
        dt: float = 1.0,
        tau_mem_init: float = 10.0,
        tau_syn_init: float = 5.0,
        threshold_init: float = 1.0,
        adapt_beta: float = 0.07,
        adapt_tau: float = 100.0,
        superspike_k: float = 25.0,
        recurrent: bool = True,
    ):
        super().__init__()
        self.size = size
        self.dt = dt
        self.superspike_k = superspike_k
        self.adapt_beta = adapt_beta

        # Learnable time constants via unconstrained parameters mapped through sigmoid
        # tau = tau_min + (tau_max - tau_min) * sigmoid(param)
        self._tau_mem_param = nn.Parameter(torch.full((size,), self._inv_sigmoid_tau(tau_mem_init)))
        self._tau_syn_param = nn.Parameter(torch.full((size,), self._inv_sigmoid_tau(tau_syn_init)))

        # Learnable base firing threshold (kept positive via softplus in forward)
        self._V_th_param = nn.Parameter(torch.full((size,), threshold_init))

        # Adaptive threshold decay factor
        self.register_buffer("adapt_decay", torch.tensor(1.0 - dt / adapt_tau))

        # Optional recurrent weight matrix
        self.recurrent = recurrent
        if recurrent:
            self.W_rec = nn.Linear(size, size, bias=False)
            nn.init.orthogonal_(self.W_rec.weight, gain=0.5)

    # --- helpers for parameterising time constants ---

    @staticmethod
    def _inv_sigmoid_tau(tau: float, tau_min: float = 1.0, tau_max: float = 50.0) -> float:
        """Inverse of the sigmoid mapping used to constrain tau."""
        import math
        x = (tau - tau_min) / (tau_max - tau_min)
        x = max(min(x, 0.999), 0.001)
        return math.log(x / (1.0 - x))

    def _get_tau(self, param: torch.Tensor, tau_min: float = 1.0, tau_max: float = 50.0) -> torch.Tensor:
        return tau_min + (tau_max - tau_min) * torch.sigmoid(param)

    @property
    def beta_mem(self) -> torch.Tensor:
        """Membrane decay factor: exp(-dt / tau_mem)."""
        tau_mem = self._get_tau(self._tau_mem_param)
        return torch.exp(-self.dt / tau_mem)

    @property
    def beta_syn(self) -> torch.Tensor:
        """Synaptic current decay factor: exp(-dt / tau_syn)."""
        tau_syn = self._get_tau(self._tau_syn_param)
        return torch.exp(-self.dt / tau_syn)

    @property
    def V_th(self) -> torch.Tensor:
        """Base firing threshold (always positive)."""
        return F.softplus(self._V_th_param)

    def forward(
        self,
        input_current: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        syn: Optional[torch.Tensor] = None,
        adapt: Optional[torch.Tensor] = None,
    ):
        """Simulate one timestep of the PLIF neuron.

        Args:
            input_current: (*, size) input current at this timestep.
            mem: membrane potential state (*, size).
            syn: synaptic current state (*, size).
            adapt: adaptive threshold variable (*, size).

        Returns:
            spike: (*, size) binary spike tensor.
            mem: updated membrane potential.
            syn: updated synaptic current.
            adapt: updated adaptive threshold variable.
        """
        if mem is None:
            mem = torch.zeros_like(input_current)
        if syn is None:
            syn = torch.zeros_like(input_current)
        if adapt is None:
            adapt = torch.zeros_like(input_current)

        beta_mem = self.beta_mem
        beta_syn = self.beta_syn
        V_th_base = self.V_th

        # Adaptive threshold: V_th_eff = V_th_base + beta_adapt * adapt
        V_th_eff = V_th_base + self.adapt_beta * adapt

        # Synaptic current dynamics
        syn = beta_syn * syn + input_current

        # Recurrent input
        rec_input = torch.zeros_like(mem)
        if self.recurrent:
            rec_input = self.W_rec(mem)

        # Membrane dynamics: leaky integration
        mem = beta_mem * mem + syn + rec_input

        # Spike generation via SuperSpike surrogate
        spike = superspike(mem, V_th_eff, self.superspike_k)

        # Soft reset: subtract threshold from membrane potential
        mem = mem - V_th_eff * spike

        # Update adaptive threshold variable
        adapt = self.adapt_decay * adapt + spike

        return spike, mem, syn, adapt


# ---------------------------------------------------------------------------
# SNNLayer: multi-sublayer spiking network with LayerNorm
# ---------------------------------------------------------------------------

class SNNLayer(nn.Module):
    """Spiking Neural Network layer with PLIF neurons, surrogate gradients,
    adaptive thresholds, recurrent connections and layer normalisation.

    Constructor:
        __init__(self, input_size, hidden_size, output_size, timesteps, dropout)

    Forward:
        x: (batch, seq_len, input_size) -> (batch, seq_len, timesteps, output_size)
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

        # Input projection
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Hidden-to-output projection
        self.fc2 = nn.Linear(hidden_size, output_size)

        # PLIF neuron populations
        self.neurons1 = PLIFNeuronLayer(hidden_size, recurrent=True)
        self.neurons2 = PLIFNeuronLayer(output_size, recurrent=True)

        # Layer normalisation between sublayers
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(output_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the spiking layer over multiple timesteps.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, timesteps, output_size).
        """
        batch_size, seq_len, _ = x.shape
        flat = batch_size * seq_len

        # Flatten batch and sequence for neuron simulation
        x_flat = x.reshape(flat, self.input_size).to(dtype=self.fc1.weight.dtype)

        # Project input once (constant current injected at every timestep)
        current1 = self.fc1(x_flat)  # (flat, hidden_size)

        # Initialise neuron states
        mem1 = torch.zeros(flat, self.hidden_size, device=x.device, dtype=current1.dtype)
        syn1 = torch.zeros_like(mem1)
        adapt1 = torch.zeros_like(mem1)

        mem2 = torch.zeros(flat, self.output_size, device=x.device, dtype=current1.dtype)
        syn2 = torch.zeros_like(mem2)
        adapt2 = torch.zeros_like(mem2)

        spikes_out = []
        for _ in range(self.timesteps):
            # First neuron layer
            spike1, mem1, syn1, adapt1 = self.neurons1(current1, mem1, syn1, adapt1)
            spike1 = self.ln1(spike1)
            spike1 = self.dropout_layer(spike1)

            # Second neuron layer
            current2 = self.fc2(spike1)
            spike2, mem2, syn2, adapt2 = self.neurons2(current2, mem2, syn2, adapt2)
            spike2 = self.ln2(spike2)

            spikes_out.append(spike2.view(batch_size, seq_len, self.output_size))

        # Stack along a new timestep dimension -> (batch, timesteps, seq_len, output_size)
        output = torch.stack(spikes_out, dim=1)
        # Permute to (batch, seq_len, timesteps, output_size) for downstream compatibility
        output = output.permute(0, 2, 1, 3)

        return output


# ---------------------------------------------------------------------------
# Temporal Attention: learn to weight timestep outputs
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Small single-head attention mechanism that produces a weighted
    combination of timestep representations.

    Input:  (batch, seq_len, timesteps, feat)
    Output: (batch, seq_len, feat)
    """

    def __init__(self, feat_dim: int, timesteps: int):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** 0.5
        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, timesteps, feat)
        Returns:
            (batch, seq_len, feat)
        """
        # Compute mean query across timesteps
        q = self.query(x.mean(dim=2))  # (batch, seq_len, feat)
        k = self.key(x)  # (batch, seq_len, timesteps, feat)

        # Attention scores: dot product between query and each timestep key
        # q: (B, S, F) -> (B, S, 1, F);  k: (B, S, T, F)
        scores = torch.einsum("bsf,bstf->bst", q, k) / self.scale  # (B, S, T)
        weights = F.softmax(scores, dim=-1)  # (B, S, T)

        # Weighted sum over timesteps
        out = torch.einsum("bst,bstf->bsf", weights, x)  # (B, S, F)
        return self.ln(out)


# ---------------------------------------------------------------------------
# MLP output head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# SNNModel: full model combining SNN, GAT, HTM with upgrades
# ---------------------------------------------------------------------------

class SNNModel(nn.Module):
    def __init__(
        self,
        sp_params: Dict,
        gat_params: Dict,
        htm_params: Dict,
        snn_params: Dict,
        device: torch.device,
    ):
        """Initialize the SNNModel.

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

        snn_input_size = sp_params["sdr_dimensions"]
        snn_hidden_size = snn_params["hidden_size"]
        snn_output_size = snn_params["output_size"]
        snn_timesteps = snn_params["timesteps"]
        snn_dropout = snn_params.get("dropout", 0.0)

        # ----- Learned input embedding (replaces naive unsqueeze+expand) -----
        input_proj_size = snn_params.get("input_size", snn_input_size)
        self.input_projection = nn.Linear(1, input_proj_size).to(device)
        self.input_ln = nn.LayerNorm(input_proj_size).to(device)

        # ----- SNN layer -----
        self.snn_layer = SNNLayer(
            input_size=input_proj_size,
            hidden_size=snn_hidden_size,
            output_size=snn_output_size,
            timesteps=snn_timesteps,
            dropout=snn_dropout,
        ).to(device)

        # ----- Temporal attention over SNN timesteps -----
        self.temporal_attention = TemporalAttention(
            feat_dim=snn_output_size,
            timesteps=snn_timesteps,
        ).to(device)

        # Layer norm after temporal attention
        self.post_snn_ln = nn.LayerNorm(snn_output_size).to(device)

        # ----- GAT layer -----
        gat_in_features = snn_output_size
        self.gat_layer = GraphAttentionLayer(
            in_features=gat_in_features,
            out_features=gat_params["out_features"],
            num_heads=gat_params["num_heads"],
            alpha=gat_params["alpha"],
            concat=self.concat,
            fallback_mode=gat_params.get("fallback_mode", "dense"),
        ).to(device)

        # Determine GAT output dimension for downstream layers
        if self.concat:
            gat_out_dim = gat_params["out_features"]
        else:
            gat_out_dim = gat_params["out_features"]

        # Layer norm after GAT
        self.post_gat_ln = nn.LayerNorm(gat_out_dim).to(device)

        # Skip connection projection (SNN -> post-GAT) when dimensions differ
        if snn_output_size != gat_out_dim:
            self.skip_snn_to_gat = nn.Linear(snn_output_size, gat_out_dim).to(device)
        else:
            self.skip_snn_to_gat = nn.Identity().to(device)

        # ----- HTM layer -----
        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        # Layer norm after HTM
        self.post_htm_ln = nn.LayerNorm(self.output_size).to(device)

        # Skip connection projection (GAT -> post-HTM) when dimensions differ
        if gat_out_dim != self.output_size:
            self.skip_gat_to_htm = nn.Linear(gat_out_dim, self.output_size).to(device)
        else:
            self.skip_gat_to_htm = nn.Identity().to(device)

        # ----- MLP output head (replaces single fc_out) -----
        mlp_hidden = max(self.output_size // 2, 64)
        self.mlp_head = MLPHead(
            in_features=self.output_size,
            hidden_features=mlp_hidden,
            out_features=self.output_size,
            dropout=snn_dropout,
        ).to(device)

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, gat_params={gat_params}, "
            f"htm_params={htm_params}, output_size={self.output_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor],
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass of the SNNModel.

        Args:
            input_ids (torch.Tensor): The input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
            adjacency_matrix (Union[scipy.sparse.csr_matrix, torch.Tensor]): The adjacency matrix.
            node_indices (torch.Tensor): The node indices of shape (batch_size,).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        # ------------------------------------------------------------------
        # 1. Learned input embedding (replaces unsqueeze+expand)
        # ------------------------------------------------------------------
        # input_ids: (batch, seq_len) -> project each scalar token id into a vector
        x = input_ids.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = self.input_projection(x)  # (batch, seq_len, input_proj_size)
        x = self.input_ln(x)

        # ------------------------------------------------------------------
        # 2. SNN layer
        # ------------------------------------------------------------------
        snn_raw = self.snn_layer(x)  # (batch, seq_len, timesteps, snn_output_size)
        logger.debug(f"SNN raw output shape: {snn_raw.shape}")

        # ------------------------------------------------------------------
        # 3. Temporal attention over timesteps
        # ------------------------------------------------------------------
        snn_output = self.temporal_attention(snn_raw)  # (batch, seq_len, snn_output_size)
        snn_output = self.post_snn_ln(snn_output)
        logger.debug(f"SNN output (after temporal attention) shape: {snn_output.shape}")

        # ------------------------------------------------------------------
        # 4. GAT layer
        # ------------------------------------------------------------------
        batch_size = snn_output.size(0)
        seq_len = snn_output.size(1)

        adj_matrix_tensor = adjacency_matrix[:seq_len, :seq_len]
        gat_output = self.gat_layer(snn_output, adj_matrix_tensor)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Residual / skip connection from SNN to GAT
        snn_skip = self.skip_snn_to_gat(snn_output)
        gat_output = self.post_gat_ln(gat_output + snn_skip)

        # ------------------------------------------------------------------
        # 5. Apply attention mask
        # ------------------------------------------------------------------
        expected_attention_mask_shape = (batch_size, seq_len)
        if attention_mask.shape != expected_attention_mask_shape:
            logger.debug(
                f"Attention mask shape {attention_mask.shape} "
                f"does not match expected {expected_attention_mask_shape}. Reshaping."
            )
            attention_mask = attention_mask.view(expected_attention_mask_shape)

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            batch_size, seq_len, gat_output.size(-1)
        )
        gat_output = gat_output * attention_mask_expanded

        # ------------------------------------------------------------------
        # 6. HTM layer
        # ------------------------------------------------------------------
        htm_output = self.htm_layer(gat_output)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Residual / skip connection from GAT to HTM
        gat_skip = self.skip_gat_to_htm(gat_output)
        htm_output = self.post_htm_ln(htm_output + gat_skip)

        # ------------------------------------------------------------------
        # 7. Select node and apply MLP output head
        # ------------------------------------------------------------------
        node_indices_expanded = node_indices.unsqueeze(-1).expand(
            batch_size, 1, htm_output.size(-1)
        )
        htm_output_selected = torch.gather(htm_output, 1, node_indices_expanded)
        htm_output_selected = htm_output_selected.squeeze(1)  # (batch, output_size)

        output = self.mlp_head(htm_output_selected)  # (batch, output_size)
        logger.debug(f"Final output shape: {output.shape}")

        # ------------------------------------------------------------------
        # 8. Store visualisation data
        # ------------------------------------------------------------------
        self.visualization_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "snn_output": snn_output,
            "gat_output": gat_output,
            "htm_output": htm_output,
            "final_output": output,
            "sdr_embeddings": self.sp_params.get("sdr_embeddings", None),
            "adjacency_matrix": adjacency_matrix,
        }

        return output

    # ------------------------------------------------------------------
    # Visualisation (simplified but present)
    # ------------------------------------------------------------------

    def visualize(self, input_ids: Optional[torch.Tensor] = None) -> None:
        """Visualize stored forward-pass data using Plotly.

        Generates heatmaps for SNN, GAT, HTM, and final outputs as well as
        output distribution violin plots.
        """
        if self.visualization_data is None:
            raise ValueError(
                "No data available for visualization. Run a forward pass first."
            )

        data = self.visualization_data
        input_ids_viz = data["input_ids"]
        snn_output = data["snn_output"]
        gat_output = data["gat_output"]
        htm_output = data["htm_output"]
        final_output = data["final_output"]

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Input Token IDs",
                "SNN Output",
                "GAT Output",
                "HTM Output",
                "Final Output",
                "Output Distribution",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        # Input token IDs heatmap
        fig.add_trace(
            go.Heatmap(z=input_ids_viz.detach().cpu().numpy(), colorscale="Viridis"),
            row=1, col=1,
        )

        # SNN output heatmap (first sample, mean over last dim if needed)
        snn_np = snn_output.detach().cpu().numpy()
        if snn_np.ndim == 3:
            snn_np = snn_np[0]
        fig.add_trace(
            go.Heatmap(z=snn_np, colorscale="Plasma"),
            row=1, col=2,
        )

        # GAT output heatmap
        gat_np = gat_output.detach().cpu().numpy()
        if gat_np.ndim == 3:
            gat_np = gat_np[0]
        fig.add_trace(
            go.Heatmap(z=gat_np, colorscale="Viridis"),
            row=1, col=3,
        )

        # HTM output heatmap
        htm_np = htm_output.detach().cpu().numpy()
        if htm_np.ndim == 3:
            htm_np = htm_np[0]
        fig.add_trace(
            go.Heatmap(z=htm_np, colorscale="Inferno"),
            row=2, col=1,
        )

        # Final output heatmap
        fig.add_trace(
            go.Heatmap(z=final_output.detach().cpu().numpy(), colorscale="Viridis"),
            row=2, col=2,
        )

        # Output distribution (violin per sample)
        final_np = final_output.detach().cpu().numpy()
        for i in range(min(final_np.shape[0], 10)):
            fig.add_trace(
                go.Violin(y=final_np[i], name=f"Sample {i}", box_visible=True),
                row=2, col=3,
            )

        fig.update_layout(
            title="SNNModel Architecture & Data Flow",
            height=800,
            width=1200,
            showlegend=False,
        )

        try:
            fig.show()
        except Exception:
            logger.warning("Could not display figure interactively.")

        try:
            fig.write_image(".visualizations/model_architecture.png")
        except Exception:
            logger.warning("Could not write visualization image to disk.")
