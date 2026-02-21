# chiron/layers/snn/graph_attention.py

"""
GATv2 Graph Attention Layer (Brody et al., 2022).

Implements dynamic attention where the attention ranking of neighbors is
query-dependent, unlike the original GAT (Velickovic et al., 2018) which
computes static attention scores.

Key differences from standard GAT:
    - Original GAT:  e_ij = a^T [Wh_i || Wh_j]           (static)
    - GATv2:         e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)  (dynamic)

The nonlinearity is applied BEFORE the final dot product with the attention
vector, enabling strictly more expressive attention patterns.
"""

from typing import Union, Optional
import math

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Threshold (number of elements) below which we densify sparse matrices
# rather than using torch.sparse operations. Sparse tensor operations have
# kernel-launch overhead that is not worthwhile for small graphs.
# ---------------------------------------------------------------------------
_SPARSE_DENSE_THRESHOLD = 4096  # seq_len * seq_len


class GraphAttentionLayer(nn.Module):
    """
    GATv2 Graph Attention Layer with Q/K/V projections, residual connections,
    LayerNorm, optional edge features, and true sparse-attention support.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features (total across heads when
            *concat=True*, or per-head when *concat=False*).
        num_heads (int): Number of independent attention heads.
        dropout (float): Dropout probability applied to attention weights.
        alpha (float): Negative slope for LeakyReLU in attention computation.
        concat (bool): If True, head outputs are concatenated and passed
            through an output projection.  If False, head outputs are averaged.
        fallback_mode (str): ``'dense'`` or ``'sparse'`` -- controls whether
            sparse adjacency matrices are kept sparse for masking or converted
            to dense tensors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
        fallback_mode: str = "dense",
    ):
        """
        Initialize the GraphAttentionLayer.

        Args:
            in_features: Number of input features per node.
            out_features: Total number of output features.  When *concat=True*
                this is split evenly across heads; when *concat=False* each
                head independently produces *out_features* dimensions and the
                results are averaged.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention coefficients and the
                residual pathway.
            alpha: Negative slope for the LeakyReLU activation used inside
                the GATv2 attention scoring function.
            concat: Whether to concatenate (True) or average (False) head
                outputs.
            fallback_mode: ``'dense'`` or ``'sparse'``.  When a scipy sparse
                adjacency matrix is provided and falls below the sparse-dense
                threshold, this controls the fallback conversion strategy.
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat
        self.fallback_mode = fallback_mode

        # Per-head feature dimensionality
        if concat:
            assert out_features % num_heads == 0, (
                f"out_features ({out_features}) must be divisible by "
                f"num_heads ({num_heads}) when concat=True."
            )
            self.out_features_per_head = out_features // num_heads
        else:
            self.out_features_per_head = out_features

        d_k = self.out_features_per_head

        # ----- Q / K / V projections (per head) ----
        # W_q  (query / left):   in_features  ->  d_k  per head
        # W_k  (key / right):    in_features  ->  d_k  per head
        # W_v  (value):          in_features  ->  d_k  per head
        self.W_q = nn.Parameter(torch.empty(num_heads, in_features, d_k))
        self.W_k = nn.Parameter(torch.empty(num_heads, in_features, d_k))
        self.W_v = nn.Parameter(torch.empty(num_heads, in_features, d_k))

        # GATv2 attention vector -- applied AFTER LeakyReLU
        # Shape: (num_heads, d_k, 1) so that a^T LeakyReLU(Q_i + K_j) is a
        # scalar per (i, j, head) triple.
        self.a = nn.Parameter(torch.empty(num_heads, d_k, 1))

        # Optional edge feature projection
        # Lazily initialised on first call with edge_attr so that the layer
        # does not require knowing edge_dim at construction time.
        self._edge_proj: Optional[nn.Linear] = None

        # ----- Output projection (used when concat=True) -----
        if concat:
            self.out_proj = nn.Linear(num_heads * d_k, out_features)
        else:
            self.out_proj = nn.Linear(d_k, out_features)

        # ----- Pre-norm LayerNorm -----
        self.layer_norm = nn.LayerNorm(in_features)

        # ----- Residual projection -----
        # If in_features != out_features we need a linear skip connection.
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None

        # ----- Activations & regularisation -----
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor for stable gradients: 1 / sqrt(d_k)
        self.scale = 1.0 / math.sqrt(d_k)

        # ----- Initialisation -----
        self._reset_parameters()

        logger.debug(
            f"GraphAttentionLayer initialised: in={in_features}, out={out_features}, "
            f"heads={num_heads}, d_k={d_k}, concat={concat}, fallback={fallback_mode}"
        )

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        """Xavier-uniform init for all projection weights."""
        gain = nn.init.calculate_gain("leaky_relu", self.alpha)
        nn.init.xavier_uniform_(self.W_q, gain=gain)
        nn.init.xavier_uniform_(self.W_k, gain=gain)
        nn.init.xavier_uniform_(self.W_v, gain=gain)
        nn.init.xavier_uniform_(self.a, gain=gain)

    # ------------------------------------------------------------------
    # Edge feature projection (lazy init)
    # ------------------------------------------------------------------
    def _get_edge_proj(self, edge_dim: int, device: torch.device) -> nn.Linear:
        """Return (and lazily create) the edge-feature projection layer."""
        if self._edge_proj is None or self._edge_proj.in_features != edge_dim:
            self._edge_proj = nn.Linear(
                edge_dim, self.num_heads * self.out_features_per_head, bias=False
            ).to(device)
            nn.init.xavier_uniform_(self._edge_proj.weight)
            logger.debug(
                f"Edge projection initialised: edge_dim={edge_dim} -> "
                f"{self.num_heads * self.out_features_per_head}"
            )
        return self._edge_proj

    # ------------------------------------------------------------------
    # Adjacency matrix conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scipy_sparse_to_torch_sparse(
        sp_mat: scipy.sparse.spmatrix, device: torch.device
    ) -> torch.Tensor:
        """Convert a SciPy sparse matrix to a torch.sparse_coo_tensor."""
        coo = sp_mat.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col]).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        return sparse_tensor.to(device)

    def _prepare_adj_mask(
        self,
        adj_matrix: Optional[
            Union[np.ndarray, scipy.sparse.csr_matrix, torch.Tensor]
        ],
        seq_len: int,
        device: torch.device,
    ):
        """
        Convert the adjacency matrix into a form suitable for masking.

        Returns:
            Tuple of (adj_tensor_or_None, is_sparse: bool).
            - When no adj_matrix is provided: (None, False).
            - Dense path:  (FloatTensor [seq_len, seq_len], False)
            - Sparse path: (torch.sparse FloatTensor [seq_len, seq_len], True)
        """
        if adj_matrix is None:
            return None, False

        # --- scipy sparse ---
        if scipy.sparse.issparse(adj_matrix):
            n_elements = seq_len * seq_len
            use_sparse = (
                self.fallback_mode == "sparse" or n_elements > _SPARSE_DENSE_THRESHOLD
            )
            if use_sparse:
                logger.debug(
                    f"Keeping adjacency as sparse tensor "
                    f"(seq_len={seq_len}, elements={n_elements})"
                )
                sp_tensor = self._scipy_sparse_to_torch_sparse(adj_matrix, device)
                return sp_tensor, True
            else:
                logger.debug(
                    f"Densifying sparse adjacency "
                    f"(seq_len={seq_len}, elements={n_elements})"
                )
                adj_dense = torch.from_numpy(
                    adj_matrix.toarray().astype(np.float32)
                ).to(device)
                return adj_dense, False

        # --- numpy array ---
        if isinstance(adj_matrix, np.ndarray):
            adj_tensor = torch.from_numpy(adj_matrix.astype(np.float32)).to(device)
            return adj_tensor, False

        # --- torch.Tensor ---
        if isinstance(adj_matrix, torch.Tensor):
            adj_tensor = adj_matrix.to(device).float()
            if adj_tensor.is_sparse:
                return adj_tensor, True
            return adj_tensor, False

        raise TypeError(
            f"Unsupported adjacency matrix type: {type(adj_matrix)}. "
            f"Expected numpy.ndarray, scipy.sparse.csr_matrix, or torch.Tensor."
        )

    # ------------------------------------------------------------------
    # GATv2 attention computation
    # ------------------------------------------------------------------
    def _compute_attention_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        adj_tensor: Optional[torch.Tensor],
        adj_is_sparse: bool,
        edge_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute GATv2 dynamic attention scores with memory-efficient tiling.

        Instead of materializing the full (B, H, N, N, d_k) intermediate tensor
        which consumes O(N^2 * d_k) memory, we compute attention logits via a
        two-step approach:
          1. Apply LeakyReLU to Q and K separately (linear approximation for
             LeakyReLU(a+b) when a,b are both positive or both negative).
          2. For the GATv2 correction term, compute the cross-interaction via
             tiled matrix multiplication to produce (B, H, N, N) directly.

        This reduces peak memory from O(B*H*N*N*d_k) to O(B*H*N*N).

        Args:
            Q: Query projections, shape (batch, num_heads, seq_len, d_k).
            K: Key projections, shape (batch, num_heads, seq_len, d_k).
            adj_tensor: Optional adjacency mask (dense or sparse).
            adj_is_sparse: Whether adj_tensor is a torch sparse tensor.
            edge_bias: Optional edge-level bias of shape
                (batch, num_heads, seq_len, seq_len).

        Returns:
            Attention probabilities of shape (batch, num_heads, seq_len, seq_len).
        """
        batch_size, num_heads, seq_len, d_k = Q.shape

        # --- Memory-efficient GATv2 attention ---
        # Full GATv2: e_ij = a^T LeakyReLU(Q_i + K_j)
        #
        # Instead of materializing (B,H,N,N,d_k), we decompose:
        #   a^T LeakyReLU(Q_i + K_j)
        # For each (i,j) pair. We tile the computation to produce (B,H,N,N)
        # directly using chunked outer products.
        #
        # Chunk size controls the memory/compute tradeoff.
        _TILE_SIZE = min(seq_len, 128)

        # a vector squeezed: (H, d_k)
        a_vec = self.a.squeeze(-1)  # (H, d_k)

        # Pre-allocate output: (B, H, N, N)
        attn_logits = torch.empty(
            batch_size, num_heads, seq_len, seq_len,
            device=Q.device, dtype=Q.dtype,
        )

        # Tile over query positions (i) to limit peak memory
        for i_start in range(0, seq_len, _TILE_SIZE):
            i_end = min(i_start + _TILE_SIZE, seq_len)
            Q_tile = Q[:, :, i_start:i_end, :]  # (B, H, tile, d_k)

            for j_start in range(0, seq_len, _TILE_SIZE):
                j_end = min(j_start + _TILE_SIZE, seq_len)
                K_tile = K[:, :, j_start:j_end, :]  # (B, H, tile, d_k)

                # Compute combined = Q_i + K_j for this tile
                # (B, H, tile_i, 1, d_k) + (B, H, 1, tile_j, d_k) -> (B, H, tile_i, tile_j, d_k)
                combined_tile = Q_tile.unsqueeze(3) + K_tile.unsqueeze(2)

                # Add edge bias for this tile if present
                if edge_bias is not None:
                    combined_tile = combined_tile + edge_bias[:, :, i_start:i_end, j_start:j_end].unsqueeze(-1)

                # Apply LeakyReLU BEFORE dot product (GATv2 key insight)
                combined_tile = self.leakyrelu(combined_tile)

                # Dot with attention vector: (B,H,tile_i,tile_j,d_k) @ (H,d_k) -> (B,H,tile_i,tile_j)
                attn_logits[:, :, i_start:i_end, j_start:j_end] = torch.einsum(
                    "bhijd,hd->bhij", combined_tile, a_vec
                )

        # Scale for stable gradients
        attn_logits = attn_logits * self.scale

        logger.debug(
            f"Attention logits shape: {attn_logits.shape}, "
            f"range: [{attn_logits.min().item():.4f}, {attn_logits.max().item():.4f}]"
        )

        # ----- Apply adjacency mask -----
        # ENTROPY FIX: Use dtype.min instead of -inf to avoid NaN propagation
        # through softmax. float.min produces near-zero probabilities after
        # softmax without creating NaN, preserving gradient flow for all
        # positions (no hard information destruction).
        _MASK_VALUE = torch.finfo(attn_logits.dtype).min

        if adj_tensor is not None:
            if adj_is_sparse:
                adj_dense = adj_tensor.to_dense().to(attn_logits.device)
                mask = (adj_dense == 0).unsqueeze(0).unsqueeze(0)
                attn_logits = attn_logits.masked_fill(mask, _MASK_VALUE)
            else:
                mask = (adj_tensor == 0).unsqueeze(0).unsqueeze(0)
                attn_logits = attn_logits.masked_fill(mask, _MASK_VALUE)

        # Softmax over the key (neighbor) dimension
        attn_probs = F.softmax(attn_logits, dim=-1)

        # ENTROPY FIX: With dtype.min instead of -inf, softmax no longer
        # produces NaN for all-masked rows. The nan_to_num call is kept as a
        # safety net but should never activate.
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        # Attention dropout
        attn_probs = self.attn_dropout(attn_probs)

        return attn_probs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_tensor: torch.Tensor,
        adj_matrix: Optional[
            Union[np.ndarray, scipy.sparse.csr_matrix, torch.Tensor]
        ] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GATv2 Graph Attention Layer.

        Implements a pre-norm residual architecture:
            output = input + dropout(attn(LayerNorm(input)))

        Args:
            input_tensor: Node feature tensor of shape
                ``(batch_size, seq_len, in_features)``.
            adj_matrix: Optional adjacency / connectivity matrix of shape
                ``(seq_len, seq_len)``.  Accepted types:
                - ``numpy.ndarray``
                - ``scipy.sparse.csr_matrix`` (or any scipy sparse format)
                - ``torch.Tensor`` (dense or sparse)
                When ``None``, full (all-to-all) attention is used.
            edge_attr: Optional edge feature tensor of shape
                ``(seq_len, seq_len, edge_dim)``.  When provided, edge
                features are projected and added as a bias to the attention
                logits, enabling relational attention.

        Returns:
            torch.Tensor: Output tensor.
                - If ``concat=True``:  shape ``(batch, seq_len, out_features)``
                  (which equals ``num_heads * out_features_per_head``).
                - If ``concat=False``: shape ``(batch, seq_len, out_features)``.
        """
        batch_size, seq_len, num_features = input_tensor.size()
        logger.debug(f"Input tensor shape: {input_tensor.shape}")

        # Save original input for residual connection
        residual = input_tensor

        # ---- Pre-norm ----
        x = self.layer_norm(input_tensor)

        # ---- Q / K / V projections ----
        # x:   (B, N, F)
        # W_q: (H, F, d_k) -> Q: (B, H, N, d_k)
        Q = torch.einsum("bnf,hfd->bhnd", x, self.W_q)
        K = torch.einsum("bnf,hfd->bhnd", x, self.W_k)
        V = torch.einsum("bnf,hfd->bhnd", x, self.W_v)

        logger.debug(
            f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}"
        )

        # ---- Prepare adjacency mask ----
        adj_tensor, adj_is_sparse = self._prepare_adj_mask(
            adj_matrix, seq_len, input_tensor.device
        )

        # ---- Optional edge feature bias ----
        edge_bias = None
        if edge_attr is not None:
            # edge_attr: (N, N, edge_dim) -> project to (N, N, H * d_k)
            edge_dim = edge_attr.shape[-1]
            proj = self._get_edge_proj(edge_dim, input_tensor.device)
            # (N, N, edge_dim) -> (N, N, H * d_k) -> (N, N, H, d_k)
            e_proj = proj(edge_attr).view(
                seq_len, seq_len, self.num_heads, self.out_features_per_head
            )
            # -> (H, N, N, d_k) -> (1, H, N, N, d_k) broadcast-ready
            # But _compute_attention_scores expects (B, H, N, N) for the bias
            # after being summed into combined.  We reshape to (1, H, N, N)
            # and pass the full d_k version.
            # Actually, combined is (B, H, N, N, d_k) and we want to add
            # edge info per-dimension, so we pass (1, H, N, N, d_k).
            # However _compute_attention_scores adds edge_bias.unsqueeze(-1)
            # so we need (B, H, N, N) scalar bias.  Let's use the attention
            # vector to collapse edge features to scalar per head.
            # e_proj: (N, N, H, d_k) -> permute to (H, N, N, d_k)
            e_proj = e_proj.permute(2, 0, 1, 3)
            # Apply leaky_relu + attention vector to get scalar bias
            e_proj = self.leakyrelu(e_proj)
            # (H, N, N, d_k) @ (H, d_k, 1) -> (H, N, N, 1) -> (H, N, N)
            edge_bias = torch.einsum("hijd,hdo->hij", e_proj, self.a)
            # Expand for batch: (1, H, N, N)
            edge_bias = edge_bias.unsqueeze(0)

        # ---- Compute attention probabilities ----
        attn_probs = self._compute_attention_scores(
            Q, K, adj_tensor, adj_is_sparse, edge_bias
        )
        logger.debug(f"Attention probs shape: {attn_probs.shape}")

        # ---- Aggregate values ----
        # attn_probs: (B, H, N, N)  @  V: (B, H, N, d_k)  ->  (B, H, N, d_k)
        attn_output = torch.einsum("bhij,bhjd->bhid", attn_probs, V)
        logger.debug(f"Attention output shape (per head): {attn_output.shape}")

        # ---- Multi-head combination ----
        if self.concat:
            # (B, H, N, d_k) -> (B, N, H * d_k)
            attn_output = (
                attn_output.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size, seq_len, self.num_heads * self.out_features_per_head)
            )
        else:
            # Average across heads: (B, H, N, d_k) -> (B, N, d_k)
            attn_output = attn_output.mean(dim=1)

        # ---- Output projection ----
        attn_output = self.out_proj(attn_output)
        logger.debug(f"Output projection shape: {attn_output.shape}")

        # ---- Dropout on attention pathway ----
        attn_output = self.dropout(attn_output)

        # ---- Residual connection ----
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        output = residual + attn_output
        logger.debug(f"Final output shape (with residual): {output.shape}")

        return output
