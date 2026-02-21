# chiron/layers/htm/model.py
#
# Fully differentiable HTM Spatial Pooler in PyTorch.
# All computation is done on GPU-compatible tensors -- no numpy, no CPU round-trips.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Straight-Through Estimator helpers
# ---------------------------------------------------------------------------

class _STEThreshold(torch.autograd.Function):
    """Hard threshold in the forward pass, straight-through gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Pass gradient through unchanged (straight-through estimator).
        return grad_output


class _STETopK(torch.autograd.Function):
    """Top-k binary mask in forward, straight-through gradient in backward."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: (N, C)  ->  mask: (N, C) with exactly k ones per row
        _, topk_indices = scores.topk(k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # Straight-through: pass gradient to all positions.
        return grad_output, None


def _ste_threshold(x: torch.Tensor) -> torch.Tensor:
    """Apply hard threshold at 0 with straight-through estimator."""
    return _STEThreshold.apply(x)


def _ste_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k positions with straight-through estimator."""
    return _STETopK.apply(scores, k)


# ---------------------------------------------------------------------------
# HTMSpatialPooler  -- fully differentiable, pure PyTorch
# ---------------------------------------------------------------------------

class HTMSpatialPooler(nn.Module):
    """
    Hierarchical Temporal Memory (HTM) Spatial Pooler -- differentiable PyTorch edition.

    Key design choices:
      * Permanences are stored as an ``nn.Parameter`` so they participate in
        autograd and can optionally be tuned by an external optimiser.
      * The binary *connected* mask is produced by a **sigmoid approximation**
        of the step function at ``syn_perm_connected`` with a configurable
        sharpness (``sigmoid_sharpness``).  Gradients flow through the sigmoid
        in the backward pass (straight-through estimator on top for the final
        binarisation when needed).
      * Overlap, boosting, inhibition, and Hebbian permanence updates are
        fully vectorised (no Python loops over columns or batch elements).
      * Duty-cycle tracking uses an exponential moving average (EMA).
    """

    def __init__(
        self,
        input_size: int,
        minicolumn_size: int,
        potential_radius: int,
        potential_pct: float,
        global_inhibition: bool,
        local_area_density: float,
        num_active_columns_per_inhibition_area: int,
        stimulus_threshold: float,
        syn_perm_inactive_dec: float,
        syn_perm_active_inc: float,
        syn_perm_connected: float,
        min_pct_overlap_duty_cycle: float,
        duty_cycle_period: int,
        max_boost: float,
        seed: int,
        sigmoid_sharpness: float = 10.0,
        **extra_kwargs,
    ):
        super().__init__()

        # Warn about (and ignore) unrecognised keyword arguments so that
        # call-sites that forward ``seq_len`` or similar do not crash.
        if extra_kwargs:
            logger.debug(
                f"HTMSpatialPooler ignoring unrecognised kwargs: "
                f"{list(extra_kwargs.keys())}"
            )

        # ---- hyper-parameters --------------------------------------------------
        self.input_size = input_size
        self.minicolumn_size = minicolumn_size
        self.num_minicolumns = (input_size + minicolumn_size - 1) // minicolumn_size
        self.potential_radius = potential_radius
        self.potential_pct = potential_pct
        self.global_inhibition = global_inhibition
        self.local_area_density = local_area_density
        self.num_active_columns_per_inhibition_area = num_active_columns_per_inhibition_area
        self.stimulus_threshold = stimulus_threshold
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_connected = syn_perm_connected
        self.min_pct_overlap_duty_cycle = min_pct_overlap_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.max_boost = max_boost
        self.seed = seed
        self.sigmoid_sharpness = sigmoid_sharpness

        # ---- derived constants -------------------------------------------------
        # EMA decay factor:  alpha  s.t.  effective window ~ duty_cycle_period
        self._ema_alpha: float = 2.0 / (duty_cycle_period + 1.0)
        self._num_active: int = min(
            num_active_columns_per_inhibition_area, self.num_minicolumns
        )

        # ---- deterministic initialisation with the given seed ------------------
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Potential synapses mask  (num_minicolumns, input_size)
        # Each column samples ``potential_pct`` fraction of inputs within
        # ``potential_radius`` of its "centre".
        potential_mask = self._init_potential_mask(gen)
        self.register_buffer("potential_mask", potential_mask)

        # Permanences initialised around the connected threshold.
        init_perm = torch.empty(self.num_minicolumns, input_size)
        init_perm.uniform_(
            syn_perm_connected - 0.05,
            syn_perm_connected + 0.05,
            generator=gen,
        )
        # Zero out non-potential synapses.
        init_perm = init_perm * potential_mask
        init_perm = init_perm.clamp(0.0, 1.0)
        self.permanences = nn.Parameter(init_perm)

        # ---- buffers for homeostatic state (not learnable) ---------------------
        self.register_buffer(
            "active_duty_cycles",
            torch.zeros(self.num_minicolumns),
        )
        self.register_buffer(
            "overlap_duty_cycles",
            torch.zeros(self.num_minicolumns),
        )
        self.register_buffer(
            "boosting_factors",
            torch.ones(self.num_minicolumns),
        )
        self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_potential_mask(self, gen: torch.Generator) -> torch.Tensor:
        """Build a binary mask of potential synapses for every minicolumn."""
        mask = torch.zeros(self.num_minicolumns, self.input_size)
        for col in range(self.num_minicolumns):
            # Determine the "centre" input index for this column.
            centre = int(col * self.input_size / self.num_minicolumns)
            half_r = self.potential_radius // 2
            lo = max(0, centre - half_r)
            hi = min(self.input_size, centre + half_r + 1)
            n_potential = max(1, int((hi - lo) * self.potential_pct))
            # Randomly select ``n_potential`` indices in [lo, hi).
            indices = torch.randperm(hi - lo, generator=gen)[:n_potential] + lo
            mask[col, indices] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Differentiable connected mask
    # ------------------------------------------------------------------

    def _connected_mask(self) -> torch.Tensor:
        """Soft approximation of the binary connected mask.

        Uses a steep sigmoid centred at ``syn_perm_connected``.  In the
        backward pass the sigmoid provides smooth gradients; in the forward
        pass we further binarise via a straight-through estimator so that
        downstream overlap is computed with exact 0/1 connections.
        """
        soft = torch.sigmoid(
            self.sigmoid_sharpness * (self.permanences - self.syn_perm_connected)
        )
        # STE: hard threshold forward, sigmoid gradient backward.
        hard = _ste_threshold(self.permanences - self.syn_perm_connected)
        # Detach the hard part and add the soft part for gradient flow.
        return hard.detach() + soft - soft.detach()

    # ------------------------------------------------------------------
    # Core forward logic
    # ------------------------------------------------------------------

    def compute_overlap(self, x: torch.Tensor) -> torch.Tensor:
        """Compute overlap scores between input and connected synapses.

        Args:
            x: (N, input_size)

        Returns:
            overlap: (N, num_minicolumns)
        """
        connected = self._connected_mask() * self.potential_mask
        # overlap_ij = sum_k  x_{i,k} * connected_{j,k}
        overlap = torch.mm(x, connected.t())  # (N, num_minicolumns)
        return overlap

    def inhibit_columns(self, boosted_overlap: torch.Tensor) -> torch.Tensor:
        """Top-k inhibition with straight-through estimator.

        Args:
            boosted_overlap: (N, num_minicolumns)

        Returns:
            active_mask: (N, num_minicolumns) -- binary {0, 1}
        """
        if self.global_inhibition:
            active_mask = _ste_topk(boosted_overlap, self._num_active)
        else:
            raise NotImplementedError("Local inhibition is not yet implemented.")
        return active_mask

    def _update_homeostasis(
        self,
        overlap: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """EMA duty-cycle update and exponential boosting (in-place, no grad).

        Called during training only.
        """
        alpha = self._ema_alpha
        n = overlap.shape[0]  # number of vectors in this batch

        # Mean activity across the batch dimension.
        batch_active_rate = active_mask.detach().mean(dim=0)          # (C,)
        batch_overlap_rate = (overlap.detach() > 0).float().mean(0)   # (C,)

        # EMA updates
        self.active_duty_cycles.mul_(1.0 - alpha).add_(alpha * batch_active_rate)
        self.overlap_duty_cycles.mul_(1.0 - alpha).add_(alpha * batch_overlap_rate)
        self._step_count.add_(1)

        # Target duty cycle
        target_density = self.active_duty_cycles.mean().clamp(min=1e-6)

        # Exponential boost: columns that fire less than the target get boosted.
        self.boosting_factors = torch.exp(
            self.max_boost * (target_density - self.active_duty_cycles)
        ).clamp(1.0, self.max_boost)

    def _hebbian_permanence_update(
        self,
        active_mask: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
        """Vectorised Hebbian permanence update (batched outer-product).

        For every active column *c* and every input position *i*:
          - if x_i == 1 (active input):   perm[c, i] += syn_perm_active_inc
          - else:                         perm[c, i] -= syn_perm_inactive_dec

        We accumulate the mean update across the batch so that permanences do
        not explode with large batch sizes.
        """
        with torch.no_grad():
            # active_mask: (N, C),  x: (N, input_size)
            # We want a (C, input_size) update averaged over the batch.

            # For active columns:  delta_pos = active_inc * x,  delta_neg = -inactive_dec * (1 - x)
            # Weight by per-sample column activity.
            # active_mask.t() @ x  -> (C, input_size): how many times each (col, input) pair co-activated
            n = x.shape[0]
            coactive = torch.mm(active_mask.t(), x) / n              # (C, input_size)
            active_count = active_mask.sum(dim=0, keepdim=True).t() / n  # (C, 1)

            inc = self.syn_perm_active_inc * coactive
            dec = self.syn_perm_inactive_dec * (active_count - coactive)

            delta = inc - dec  # (C, input_size)

            # Only modify potential synapses.
            delta = delta * self.potential_mask

            # Fused add + clamp: single kernel launch instead of two separate
            # kernel invocations (add_ then clamp_). This reduces CUDA kernel
            # launch overhead and improves memory bandwidth utilization.
            self.permanences.data = (self.permanences.data + delta).clamp_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the spatial pooler on a flat batch of input vectors.

        Args:
            x: (N, input_size)   -- values in [0, 1] (or arbitrary real-valued)

        Returns:
            active_columns: (N, num_minicolumns) -- binary {0, 1}, differentiable.
        """
        overlap = self.compute_overlap(x)                              # (N, C)
        boosted_overlap = overlap * self.boosting_factors.unsqueeze(0)  # (N, C)
        active_mask = self.inhibit_columns(boosted_overlap)            # (N, C)

        # Homeostatic updates (only during training, no autograd).
        if self.training:
            self._update_homeostasis(overlap, active_mask)
            self._hebbian_permanence_update(active_mask, x)

        return active_mask


# ---------------------------------------------------------------------------
# HTMModel  -- wraps the spatial pooler for sequence-level processing
# ---------------------------------------------------------------------------

class HTMModel(nn.Module):
    """High-level wrapper that:

    1. Projects the input to the spatial-pooler's expected ``input_size``.
    2. Runs the differentiable spatial pooler per time-step.
    3. Projects the sparse column activations to ``sdr_dimensions``.
    4. Adds a **residual connection** (through a matching linear projection
       when dimensions differ) and applies **LayerNorm** for training
       stability.
    """

    def __init__(self, sdr_dimensions: int, device: torch.device, **kwargs):
        """
        Initialise the HTMModel.

        Args:
            sdr_dimensions: Output dimensionality (SDR width).
            device: Target device (e.g. ``torch.device('cuda')``).
            **kwargs: All HTM spatial-pooler hyper-parameters.  See
                :class:`HTMSpatialPooler` for the full list.
        """
        super().__init__()

        self.output_size = sdr_dimensions
        self.device = device

        # Build the spatial pooler (extra kwargs like ``seq_len`` are
        # harmlessly absorbed by HTMSpatialPooler.__init__).
        self.spatial_pooler = HTMSpatialPooler(**kwargs)

        sp_input_size: int = self.spatial_pooler.input_size
        num_cols: int = self.spatial_pooler.num_minicolumns

        # Input projection:  arbitrary feature dim -> sp_input_size
        self.input_proj = nn.Linear(sp_input_size, sp_input_size)

        # Output projection:  num_minicolumns -> sdr_dimensions
        self.fc = nn.Linear(num_cols, sdr_dimensions)

        # Residual projection (used when input dim != output dim).
        if sp_input_size != sdr_dimensions:
            self.residual_proj: Optional[nn.Module] = nn.Linear(
                sp_input_size, sdr_dimensions
            )
        else:
            self.residual_proj = None

        # Layer normalisation for training stability.
        self.layer_norm = nn.LayerNorm(sdr_dimensions)

        # Move everything to the target device.
        self.to(device)

        logger.info(
            f"HTMModel initialised: input_size={sp_input_size}, "
            f"num_minicolumns={num_cols}, sdr_dimensions={sdr_dimensions}, "
            f"device={device}"
        )

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Process a batched sequence through the HTM spatial pooler.

        Args:
            input_sequence: ``(batch, seq_len, input_size)``

        Returns:
            ``(batch, seq_len, sdr_dimensions)``
        """
        batch_size, seq_len, input_size = input_sequence.shape
        expected_input_size = self.spatial_pooler.input_size

        # Handle mismatched input feature dimension gracefully.
        if input_size != expected_input_size:
            logger.warning(
                f"Input feature size {input_size} != expected {expected_input_size}. "
                f"Truncating / zero-padding to match."
            )
            if input_size > expected_input_size:
                input_sequence = input_sequence[:, :, :expected_input_size]
            else:
                pad = torch.zeros(
                    batch_size,
                    seq_len,
                    expected_input_size - input_size,
                    device=input_sequence.device,
                    dtype=input_sequence.dtype,
                )
                input_sequence = torch.cat([input_sequence, pad], dim=-1)

        # Save residual *before* the spatial pooler.
        residual = input_sequence  # (B, T, input_size)

        # Input projection (keeps same dim, adds learnable transform).
        x = self.input_proj(input_sequence)  # (B, T, sp_input_size)

        # Flatten batch & time, run SP, then reshape back.
        x_flat = x.reshape(batch_size * seq_len, expected_input_size)
        sp_out = self.spatial_pooler(x_flat)  # (B*T, num_minicolumns)
        sp_out = sp_out.reshape(batch_size, seq_len, self.spatial_pooler.num_minicolumns)

        # Project to SDR dimensions.
        output = self.fc(sp_out)  # (B, T, sdr_dimensions)

        # Residual connection.
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        output = output + residual

        # LayerNorm for training stability.
        output = self.layer_norm(output)

        logger.debug(
            f"HTM forward: input {input_sequence.shape} -> output {output.shape}"
        )

        return output

    def inspect(self) -> dict:
        """Return a diagnostic dictionary of internal spatial-pooler state.

        All values are returned as detached CPU tensors (or Python scalars)
        for easy inspection without interfering with autograd.
        """
        sp = self.spatial_pooler
        connected = (sp.permanences.data >= sp.syn_perm_connected).float()
        return {
            "connections": connected.detach().cpu(),
            "permanences": sp.permanences.data.detach().cpu(),
            "boosting_factors": sp.boosting_factors.detach().cpu(),
            "active_duty_cycles": sp.active_duty_cycles.detach().cpu(),
            "overlap_duty_cycles": sp.overlap_duty_cycles.detach().cpu(),
            "min_overlap_duty_cycles": (
                sp.active_duty_cycles * sp.min_pct_overlap_duty_cycle
            ).detach().cpu(),
            "min_pct_overlap_duty_cycle": sp.min_pct_overlap_duty_cycle,
            "duty_cycle_period": sp.duty_cycle_period,
            "max_boost": sp.max_boost,
            "global_inhibition": sp.global_inhibition,
            "num_active_columns_per_inhibition_area": sp.num_active_columns_per_inhibition_area,
            "potential_mask": sp.potential_mask.detach().cpu(),
            "sigmoid_sharpness": sp.sigmoid_sharpness,
            "step_count": sp._step_count.item(),
        }
