import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class FlyHashProjection(nn.Module):
    """
    Fly Hash projection inspired by the Drosophila olfactory circuit
    (Dasgupta et al., 2017 - "A neural algorithm for a fundamental computing problem").

    The fruit fly's olfactory system uses sparse random projections to expand
    dimensionality before applying winner-take-all inhibition, producing
    locality-sensitive hash codes that preserve similarity while being sparse.

    Adds learnable gain modulation on top of the fixed random projection
    for gradient-based fine-tuning of the hash.
    """

    def __init__(self, input_dim: int, expansion_dim: int, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Sparse binary random projection (~10% connectivity per neuron)
        connection_prob = min(10.0 / input_dim, 0.5)
        mask = torch.bernoulli(
            torch.full((expansion_dim, input_dim), connection_prob), generator=gen
        )
        weights = torch.randn(expansion_dim, input_dim, generator=gen) * mask
        row_norms = weights.norm(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / row_norms
        self.register_buffer("projection_weights", weights)
        self.gain = nn.Parameter(torch.ones(expansion_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.mm(x, self.projection_weights.t())
        return projected * self.gain.unsqueeze(0)


class KWinnersTakeAll(nn.Module):
    """
    k-Winners-Take-All activation for biologically plausible sparsification.

    Implements competitive inhibition where only the top-k most active neurons
    survive, with straight-through estimator for gradient flow and boosting
    for homeostatic regulation.

    References:
    - Ahmad & Hawkins (2016) - "How do neurons operate on sparse distributed representations?"
    - Makhzani & Frey (2015) - "Winner-Take-All Autoencoders"
    """

    def __init__(self, sparsity: float = 0.02, boost_strength: float = 1.5):
        super().__init__()
        self.sparsity = sparsity
        self.boost_strength = boost_strength
        self.register_buffer("duty_cycles", None)

    def _compute_k(self, n: int) -> int:
        return max(1, int(round(n * self.sparsity)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        k = self._compute_k(n)

        if self.training and self.duty_cycles is not None:
            target_duty = self.sparsity
            boost = torch.exp(
                self.boost_strength * (target_duty - self.duty_cycles)
            )
            x = x * boost.unsqueeze(0)

        topk_vals, _ = torch.topk(x, k, dim=1)
        threshold = topk_vals[:, -1:]
        winners = (x >= threshold).float()

        if self.training:
            batch_duty = winners.mean(dim=0)
            if self.duty_cycles is None:
                self.duty_cycles = batch_duty.detach()
            else:
                self.duty_cycles = (
                    0.99 * self.duty_cycles + 0.01 * batch_duty.detach()
                )

        # Straight-through estimator
        return x + (winners - x).detach()


class SDRGenerator(nn.Module):
    """
    Generates Sparse Distributed Representations (SDRs) using a bio-inspired
    pipeline: Fly Hash expansion + k-Winners-Take-All sparsification.

    Replaces the original TruncatedSVD + percentile-threshold approach with:
    1. Learnable linear projection (replaces SVD)
    2. Sparse random expansion via Fly Hash (dimensionality expansion)
    3. k-WTA competitive inhibition for exact sparsity control
    4. Gradient flow support for end-to-end joint training

    Attributes:
        projection_dimensions (int): Intermediate projection dimensionality.
        sdr_dimensions (int): Final SDR dimensionality.
        sparsity (float): Fraction of active bits in the SDR.
    """

    def __init__(
        self,
        projection_dimensions: int,
        sdr_dimensions: int,
        sparsity: float,
        use_gpu: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.projection_dimensions = projection_dimensions
        self.sdr_dimensions = sdr_dimensions
        self.sparsity = sparsity
        self.use_gpu = use_gpu
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        if self.projection_dimensions >= self.sdr_dimensions:
            raise ValueError(
                "Projection dimensions should be less than SDR dimensions."
            )

        self.input_norm = nn.LayerNorm(projection_dimensions)
        self.fly_hash = FlyHashProjection(
            input_dim=projection_dimensions,
            expansion_dim=sdr_dimensions,
            seed=seed,
        )
        self.kwta = KWinnersTakeAll(sparsity=sparsity, boost_strength=1.5)
        self._input_projector = None
        self.to(self.device)

    def _get_input_projector(self, input_dim: int) -> nn.Linear:
        if self._input_projector is None or self._input_projector.in_features != input_dim:
            self._input_projector = nn.Linear(
                input_dim, self.projection_dimensions, bias=False
            ).to(self.device)
            nn.init.trunc_normal_(
                self._input_projector.weight, std=1.0 / math.sqrt(input_dim)
            )
        return self._input_projector

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Differentiable SDR generation from embedding tensor.

        Args:
            embeddings: (N, D) tensor of high-dimensional embeddings.

        Returns:
            (N, sdr_dimensions) binary SDR tensor with controlled sparsity.
        """
        projector = self._get_input_projector(embeddings.shape[-1])
        projected = projector(embeddings)
        projected = self.input_norm(projected)
        expanded = self.fly_hash(projected)
        sdr = self.kwta(expanded)
        return sdr

    def generate_sdr_embeddings(
        self, embeddings: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Generate SDR embeddings from numpy arrays (backward-compatible API).

        Args:
            embeddings: List of high-dimensional numpy vectors.

        Returns:
            (N, sdr_dimensions) numpy array of binary SDRs, or None on failure.
        """
        if not embeddings:
            logger.warning("No embeddings provided for SDR generation.")
            return None

        valid_embeddings = [
            emb for emb in embeddings
            if isinstance(emb, np.ndarray) and emb.size > 0
            and not np.isnan(emb).any() and not np.isinf(emb).any()
        ]

        if len(valid_embeddings) == 0:
            logger.warning("No valid embeddings found after filtering.")
            return None

        logger.info(
            f"Generating SDRs for {len(valid_embeddings)} embeddings "
            f"(filtered {len(embeddings) - len(valid_embeddings)} invalid)..."
        )

        try:
            embedding_tensor = torch.tensor(
                np.vstack(valid_embeddings), dtype=torch.float32
            ).to(self.device)

            batch_size = 4096
            sdr_batches = []

            with torch.no_grad():
                for start in range(0, embedding_tensor.shape[0], batch_size):
                    end = min(start + batch_size, embedding_tensor.shape[0])
                    batch = embedding_tensor[start:end]
                    sdr_batch = self.forward(batch)
                    sdr_batch = (sdr_batch > 0.5).float()
                    sdr_batches.append(sdr_batch.cpu())

            sdr_embeddings = torch.cat(sdr_batches, dim=0).numpy()

            active_frac = sdr_embeddings.mean()
            logger.info(
                f"SDR generation complete. Shape: {sdr_embeddings.shape}, "
                f"Active fraction: {active_frac:.4f} (target: {self.sparsity:.4f})"
            )
            return sdr_embeddings

        except Exception as e:
            logger.error(f"Error during SDR generation: {e}")
            return None
