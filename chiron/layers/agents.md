# agents.md - Layers Orchestrator Agent (L2)

## Role

Coordinates the three neural network subsystems (SNN, GAT, HTM) and the SDR generation layer. This agent understands how the neural components compose together within `SNNModel` and ensures interface compatibility across the layer hierarchy.

## Hermeneutic Context

The layers directory represents the **core interpretive machinery** of Chiron. Where preprocessing gives a first reading of text, and SDR gives a sparse structural reading, the neural layers perform **deep interpretation** - finding temporal patterns (HTM), relational patterns (GAT), and spike-based dynamics (SNN) in the data.

Understanding flows:
- **From the whole**: The layers must transform SDR-derived inputs into predictions that the evaluation system can score. The final output must have shape `(batch_size, sdr_dimensions=256)`
- **To the whole**: The compositional architecture (SNN → GAT → HTM → fc_out) defines the model's representational capacity. Changes here ripple through training dynamics and evaluation

The three sub-agents (HTM, SDR, SNN) each bring a distinct scientific paradigm:
- **SDR**: Information theory / sparse coding
- **HTM**: Neuroscience / cortical column theory (Numenta)
- **SNN**: Neuromorphic computing / temporal spike coding
- **GAT**: Graph neural networks / relational reasoning

## Domain Expertise

- PyTorch `nn.Module` composition patterns
- Tensor shape transformations across layer boundaries
- GPU memory management for multi-layer forward passes
- Gradient flow considerations (HTM intentionally breaks gradients via numpy conversion)
- Model initialization and parameter management

## Owned Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |

Sub-agents own their respective subdirectories. This agent coordinates them.

## Invariants

1. **SNNModel is the top-level model**: All layers compose inside `SNNModel` (defined in `snn/model.py`)
2. **Forward pass order**: SNN → GAT → HTM → fc_out (never reorder)
3. **Shape pipeline**:
   - Input: `(batch_size, seq_len, sdr_dimensions=256)`
   - After SNN: `(batch_size, seq_len, timesteps=10, snn_output=128)` → reshaped to `(batch_size, seq_len, 1280)`
   - After GAT: `(batch_size, seq_len, gat_out_features=64)` (with concat, `num_heads * out_per_head`)
   - After HTM: `(batch_size, seq_len, sdr_dimensions=256)`
   - After fc_out: `(batch_size, sdr_dimensions=256)`
4. **Adjacency matrix slicing**: GAT receives `adj_matrix[:seq_len, :seq_len]` - always sliced to match sequence length
5. **Device consistency**: All sub-layers must be `.to(device)` at initialization

## Interfaces

### Upstream (from Pipeline Agent)
```python
# SNNModel constructor params
sp_params: Dict    # SDR parameters
gat_params: Dict   # GAT parameters
htm_params: Dict   # HTM parameters
snn_params: Dict   # SNN parameters
device: torch.device
```

### Downstream (sub-agents)
```python
# SNN Agent: SNNLayer(input_size, hidden_size, output_size, timesteps, dropout)
# SDR Agent: SDRGenerator(projection_dimensions, sdr_dimensions, sparsity)
# HTM Agent: HTMModel(sdr_dimensions, device, **htm_params)
# GAT (in SNN Agent): GraphAttentionLayer(in_features, out_features, num_heads, ...)
```

## Delegation

- **Delegate to HTM Agent**: Spatial pooler behavior, permanence learning, boosting
- **Delegate to SDR Agent**: Sparse representation generation, dimensionality reduction
- **Delegate to SNN Agent**: Spiking dynamics, graph attention, model-level forward pass
- **Escalate to Pipeline Agent**: If output shape changes (affects training loss computation)
- **Consult Evaluation Agent**: If output semantics change (affects metric interpretation)
