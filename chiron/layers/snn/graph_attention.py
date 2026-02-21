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
        Compute GATv2 dynamic attention scores.

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

        # GATv2: e_ij = a^T LeakyReLU(Q_i + K_j)
        # Broadcast:  Q_i is (B, H, N, 1, d_k),  K_j is (B, H, 1, N, d_k)
        # Result:     (B, H, N, N, d_k)
        Q_expanded = Q.unsqueeze(3)       # (B, H, N, 1, d_k)
        K_expanded = K.unsqueeze(2)       # (B, H, 1, N, d_k)
        combined = Q_expanded + K_expanded  # (B, H, N, N, d_k)

        # Add edge bias if present
        if edge_bias is not None:
            # edge_bias: (B, H, N, N) -> (B, H, N, N, 1)
            combined = combined + edge_bias.unsqueeze(-1)

        # Apply LeakyReLU BEFORE the dot product (the GATv2 key insight)
        combined = self.leakyrelu(combined)

        # Dot product with attention vector:  a^T @ combined -> scalar
        # a: (H, d_k, 1),  combined: (B, H, N, N, d_k)
        # => attn_scores: (B, H, N, N)
        attn_logits = torch.einsum("bhijd,hdo->bhij", combined, self.a).squeeze(-1)

        # Scale for stable gradients
        attn_logits = attn_logits * self.scale

        logger.debug(
            f"Attention logits shape: {attn_logits.shape}, "
            f"range: [{attn_logits.min().item():.4f}, {attn_logits.max().item():.4f}]"
        )

        # ----- Apply adjacency mask -----
        if adj_tensor is not None:
            if adj_is_sparse:
                # Convert sparse adj to dense for masking -- but only the
                # boolean mask.  For truly huge graphs a custom sparse-softmax
                # kernel would be needed; here we materialise the mask which
                # is acceptable for the sequence lengths this layer targets.
                adj_dense = adj_tensor.to_dense().to(attn_logits.device)
                # Expand to (1, 1, N, N) for broadcasting
                mask = (adj_dense == 0).unsqueeze(0).unsqueeze(0)
                attn_logits = attn_logits.masked_fill(mask, float("-inf"))
            else:
                # Dense path
                mask = (adj_tensor == 0).unsqueeze(0).unsqueeze(0)
                # Broadcast across batch and heads
                attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # Softmax over the key (neighbor) dimension
        attn_probs = F.softmax(attn_logits, dim=-1)

        # Handle NaN from all-masked rows (isolated nodes)
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
