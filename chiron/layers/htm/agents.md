# agents.md - HTM Agent (L3)

## Role

Hierarchical Temporal Memory specialist. Implements the biologically-inspired spatial pooling algorithm based on Numenta's cortical column theory. This agent manages the most neuroscience-grounded component of the pipeline.

## Hermeneutic Context

HTM represents the **biological interpretation layer** in the hermeneutic circle. While SNN provides spike-based temporal dynamics and GAT provides relational attention, HTM provides the cortical column model - minicolumns that compete, learn permanences, and maintain duty cycles just like biological neurons.

Understanding flows:
- **From the whole**: HTM receives GAT-processed features and must produce SDR-dimensional output. It sits third in the neural pipeline, seeing already-enriched representations
- **To the whole**: HTM's spatial pooling creates a sparse, competition-based representation that captures the most salient features from the attention-weighted graph output. Its non-differentiable nature (numpy computation) means it acts as a regularization boundary

Key scientific concepts this agent must understand:
- **Minicolumns**: Groups of neurons that compete via inhibition
- **Permanences**: Synapse strength values that evolve through Hebbian learning
- **Boosting**: Mechanism to ensure all minicolumns participate over time
- **Duty cycles**: Running averages that track minicolumn activation frequency
- **Spatial pooling**: Converting dense input to sparse active column representation

## Owned Files

| File | Class | Purpose |
|------|-------|---------|
| `model.py` | `HTMSpatialPooler` | Core spatial pooling algorithm (non-differentiable, numpy-based) |
| `model.py` | `HTMModel` | PyTorch wrapper that bridges numpy HTM with torch pipeline |
| `__init__.py` | - | Package marker |

## Invariants

1. **Numpy boundary**: `HTMSpatialPooler` operates entirely in numpy. Torch tensors are converted at the `HTMModel` boundary via `.cpu().numpy()` and `torch.from_numpy()`
2. **Gradient flow**: Gradients DO NOT flow through HTM. The `fc` linear layer after spatial pooling is the gradient entry point
3. **Connection matrix**: Binary (0 or 1) based on permanence threshold (`syn_perm_connected=0.2`)
4. **Permanence range**: Always clipped to `[0, 1]`
5. **Boosting range**: Clipped to `[1, max_boost=5.0]`
6. **Input reshaping**: 3D input `(batch, seq, features)` is reshaped to 2D `(batch*seq, features)` for computation, then reshaped back
7. **Minicolumn count**: `ceil(input_size / minicolumn_size)` = `ceil(128 / 16)` = 8

## Configuration (from config.json)

```json
{
    "input_size": 128,
    "minicolumn_size": 16,
    "potential_radius": 100,
    "potential_pct": 0.5,
    "global_inhibition": true,
    "local_area_density": -1.0,
    "num_active_columns_per_inhibition_area": 10,
    "stimulus_threshold": 0.0,
    "syn_perm_inactive_dec": 0.01,
    "syn_perm_active_inc": 0.1,
    "syn_perm_connected": 0.2,
    "min_pct_overlap_duty_cycle": 0.005,
    "duty_cycle_period": 500,
    "max_boost": 5.0,
    "seed": 42
}
```

## Interfaces

### Upstream (from SNN Agent via SNNModel)
```python
# HTMModel.forward receives GAT output
input_sequence: torch.Tensor  # shape (batch_size, seq_len, gat_out_features)
# Returns
output: torch.Tensor  # shape (batch_size, seq_len, sdr_dimensions)
```

### Internal Pipeline
```
input_sequence → [resize to input_size] → HTMSpatialPooler.forward()
    → compute_overlap (einsum: input @ connections.T)
    → inhibit_columns (top-k selection via global inhibition)
    → update_permanences (Hebbian learning)
    → update_duty_cycles (running average)
    → update_boosting_factors (exponential boost)
    → active_columns → fc(active_columns) → output
```

## Delegation

- **Escalate to Layers Orchestrator**: If input/output dimensions change
- **Consult SNN Agent**: HTMModel is instantiated inside SNNModel - constructor param changes need coordination
- **Consult Testing Agent**: HTM has the only existing tests (`tests/htm/pooler/`)
