# agents.md - SDR Agent (L3)

## Role

Sparse Distributed Representation generation specialist. Transforms dense Word2Vec embeddings into sparse binary vectors using dimensionality reduction (TruncatedSVD) and percentile-based thresholding. This agent owns the critical representation bottleneck of the pipeline.

## Hermeneutic Context

SDR generation is the **interpretive compression** step in the hermeneutic circle. Dense 300-dimensional Word2Vec embeddings carry rich but redundant semantic information. SDR generation interprets this information through the lens of sparsity - asking "which features are most distinctive?" and encoding only those as active bits.

Understanding flows:
- **From the whole**: The pipeline needs binary, sparse representations that HTM can process with its minicolumn architecture. SDR dimensions (256 target, 64 projection) and sparsity (0.2%) are tuned for downstream consumption
- **To the whole**: The quality of SDRs determines the information available to all downstream neural layers. SDR sparsity directly affects adjacency matrix density and HTM activation patterns

Key scientific concepts:
- **Sparse coding**: Representing information with few active elements (0.2% = ~0.5 active bits in 256)
- **Dimensionality reduction**: SVD finds principal components that capture maximum variance
- **Binary thresholding**: Percentile-based binarization preserves relative importance

## Owned Files

| File | Class | Purpose |
|------|-------|---------|
| `sdr_generation.py` | `SDRGenerator` | SVD-based dimensionality reduction + binary thresholding |
| `__init__.py` | - | Package marker |

## Invariants

1. **Projection dimensions < SDR dimensions**: `projection_dimensions` (64) must be strictly less than `sdr_dimensions` (256). Enforced by ValueError
2. **Output is binary**: SDR values are always 0 or 1 (integer) after thresholding
3. **Sparsity is global**: Threshold is computed as a global percentile across all reduced embeddings, not per-embedding
4. **Invalid embeddings are filtered**: Empty, NaN, and Inf embeddings are silently dropped
5. **TruncatedSVD is fit on all data**: `fit_transform` is called on the concatenated embedding matrix - this means SDR generation is not incremental
6. **Return type**: `np.ndarray` of shape `(num_valid_embeddings, projection_dimensions)` or `None` on failure

## Configuration (from config.json)

```json
{
    "projection_dimensions": 64,
    "sdr_dimensions": 256,
    "sparsity": 0.002,
    "use_gpu": true
}
```

Note: `use_gpu` is accepted but not used in the current implementation (SVD runs on CPU via scikit-learn).

## Interfaces

### Upstream (from Preprocessing Agent via Pipeline)
```python
# Input: Word2Vec embeddings
embeddings: List[np.ndarray]  # each shape (300,), dtype float32
```

### Downstream (to Pipeline Agent for adjacency matrix and model)
```python
# Output: SDR binary matrix
sdr_embeddings: np.ndarray  # shape (N, 64), dtype int, values {0, 1}
# Note: output shape is (N, projection_dimensions=64), NOT (N, sdr_dimensions=256)
```

### Pipeline Integration
```
Word2Vec (300-dim) → filter invalid → vstack → TruncatedSVD (→ 64-dim)
    → percentile threshold at (1 - 0.002) * 100 = 99.8th percentile
    → binary SDR → adjacency matrix computation
```

## Delegation

- **Escalate to Layers Orchestrator**: If SDR dimensionality changes (affects all downstream layers)
- **Consult Preprocessing Agent**: If embedding format changes
- **Consult Pipeline Agent**: SDR embeddings feed directly into adjacency matrix computation in `main.py`

## Technical Notes

- The actual output dimension is `projection_dimensions` (64), not `sdr_dimensions` (256). The `sdr_dimensions` param is used for validation only
- With sparsity=0.002 and projection_dimensions=64, the threshold is at the 99.8th percentile - extremely sparse output
- `use_gpu` parameter exists for future GPU-accelerated SVD but is currently unused
