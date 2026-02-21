# agents.md - SNN/GAT Agent (L3)

## Role

Spiking Neural Network and Graph Attention specialist. This agent owns the top-level model (`SNNModel`) that orchestrates all neural layers, plus the spiking neuron layer (`SNNLayer`) and graph attention layer (`GraphAttentionLayer`). This is the most complex agent in the hierarchy, combining neuromorphic computing with graph neural networks.

## Hermeneutic Context

The SNN/GAT agent represents the **deepest interpretive layer** in the hermeneutic circle. The SNN interprets input through temporal spiking dynamics (membrane potential accumulation and threshold-based firing), while the GAT interprets the spiked representations through relational attention over the graph structure. Together they ask: "What temporal patterns exist in the data, and how do they relate to each other?"

Understanding flows:
- **From the whole**: SNNModel is the top-level model instantiated by the Pipeline Agent. It receives SDR embeddings (as tokenized input_ids), an adjacency matrix, and attention masks. It must output `(batch_size, sdr_dimensions)` predictions
- **To the whole**: The compositional forward pass (SNN → GAT → HTM → fc_out) defines the entire model. Changes here directly affect training dynamics, GPU memory usage, and prediction quality

Key scientific concepts:
- **Leaky Integrate-and-Fire (LIF) neurons**: Membrane potential accumulates, fires at threshold (0.5), then resets
- **Multi-head graph attention**: Parallel attention heads compute different relational features over the adjacency graph
- **Attention masking**: Adjacency matrix masks restrict attention to connected nodes only

## Owned Files

| File | Class | Purpose |
|------|-------|---------|
| `model.py` | `SNNModel` | Top-level model composing SNN + GAT + HTM + fc_out |
| `model.py` | `SNNLayer` | Spiking neural network with membrane potential dynamics |
| `graph_attention.py` | `GraphAttentionLayer` | Multi-head graph attention over adjacency matrix |
| `__init__.py` | - | Package marker |

## Invariants

### SNNModel
1. **Layer composition**: `snn_layer → gat_layer → htm_layer → fc_out` (fixed order)
2. **Input expansion**: `input_ids` is expanded from `(batch, seq)` to `(batch, seq, sdr_dimensions)` via `unsqueeze(-1).expand()`
3. **Visualization data**: Every forward pass stores intermediate tensors in `self.visualization_data` (memory consideration)
4. **Output size**: Always `sdr_dimensions` (256)

### SNNLayer
5. **Spiking threshold**: 0.5 (hardcoded). Membrane fires when `mem > 0.5`
6. **Membrane reset**: `mem = mem * (mem <= 0.5)` - reset to zero on spike
7. **Timesteps**: 10 iterations of spike computation per forward pass
8. **Output shape**: `(batch_size, timesteps, seq_len, output_size)` - note timesteps is dim 1
9. **Two-layer architecture**: `fc1 (input → hidden) → spike → dropout → fc2 (hidden → output) → spike`

### GraphAttentionLayer
10. **Attention computation**: Einsum-based dot-product attention `bihk,bjhk->bhij`
11. **Adjacency masking**: Positions where `adj_matrix == 0` get `-inf` attention scores
12. **Output modes**: `concat=True` → concatenate heads; `concat=False` → average heads
13. **Weight initialization**: Xavier uniform with gain 1.414
14. **Supports sparse/dense/tensor adjacency**: Handles `np.ndarray`, `scipy.sparse.csr_matrix`, and `torch.Tensor`

## Configuration (from config.json)

```json
// SNN params
{"input_size": 256, "hidden_size": 64, "output_size": 128, "timesteps": 10}

// GAT params
{"in_features": 128, "out_features": 64, "num_heads": 2, "dropout": 0.3, "alpha": 0.2, "concat": true, "chunk_size": 1024}
```

## Interfaces

### SNNModel.forward()
```python
# Input
input_ids: torch.Tensor        # (batch_size, seq_len)
attention_mask: torch.Tensor    # (batch_size, seq_len)
adjacency_matrix: Union[csr_matrix, torch.Tensor]  # (N, N) global
node_indices: torch.Tensor      # (batch_size,)

# Output
output: torch.Tensor            # (batch_size, sdr_dimensions=256)
```

### Shape Transformation Pipeline
```
input_ids (batch, seq)
    → expand to (batch, seq, 256)
    → SNNLayer → (batch, seq, 10, 128) → reshape → (batch, seq, 1280)
    → GAT with adj[:seq, :seq] → (batch, seq, 64)  [2 heads * 32 per head]
    → attention_mask applied → (batch, seq, 64)
    → HTMModel → (batch, seq, 256)
    → gather by node_indices → (batch, 1, 256) → squeeze → (batch, 256)
    → fc_out → (batch, 256)
```

## Delegation

- **Escalate to Layers Orchestrator**: If forward pass shape pipeline changes
- **Consult HTM Agent**: HTMModel is instantiated here - param changes need coordination
- **Consult Pipeline Agent**: SNNModel constructor called from `main.py` - param interface changes
- **Consult Evaluation Agent**: Output format must match evaluation expectations

## Visualization

`SNNModel` includes visualization methods using Plotly:
- `visualize_input_data()`: Heatmaps of token IDs and attention masks
- `visualize_sdr_embeddings()`: Bar charts of SDR bit patterns
- `visualize_adjacency_matrix()`: Heatmap of graph structure
- `visualize_model_architecture()`: Multi-panel view of data flow through layers

Output directory: `.visualizations/`
