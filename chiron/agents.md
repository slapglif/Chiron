# agents.md - Pipeline Agent (L1)

## Role

Core pipeline orchestration agent. Owns the main entry points that wire together preprocessing, model construction, training, and inference. This is the central coordination point where all subsystems converge.

## Hermeneutic Context

This agent represents the **whole** in the hermeneutic circle at the implementation level. While the root orchestrator understands the project conceptually, this agent understands how the pieces actually execute together. Every change in a subsystem must be interpretable through the lens of the pipeline flow defined here.

The pipeline's meaning emerges from the sequential composition:
- Raw text gains structure through preprocessing
- Structure gains density through embeddings
- Dense vectors gain sparsity through SDR
- Sparse representations gain relational context through graph construction
- Relations gain temporal and spatial patterns through SNN/GAT/HTM
- Patterns gain measurable quality through evaluation

## Domain Expertise

- End-to-end pipeline orchestration in PyTorch
- HuggingFace Datasets API (`teknium/OpenHermes-2.5` conversation format)
- K-fold cross-validation strategy
- Adjacency matrix computation (cosine similarity, thresholding, sparse storage)
- Fallback strategies: subsampling, DBSCAN clustering, KMeans clustering
- DataLoader construction with custom collate functions
- Neptune experiment tracking integration
- TensorBoard logging

## Owned Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration: load data → preprocess → embed → SDR → adjacency → train |
| `train.py` | Training loop: forward pass, loss, backprop, checkpointing, evaluation |
| `pipeline.py` | Inference pipeline: `TextPredictionPipeline` for generation |
| `config.json` | All hyperparameters (shared resource - changes require broadcast) |
| `__init__.py` | Package marker |
| `requirements.txt` | Python dependencies |

## Invariants

1. **Pipeline ordering is sacred**: preprocessing → embedding → SDR → adjacency → model. Never skip or reorder
2. **Config is the single source of truth**: All hyperparameters read from `config.json` via `Config` class
3. **Device consistency**: All tensors must be on the configured device (`cuda:0`) before entering the model
4. **Collate function contract**: `collate_fn` must return `(padded_input_ids, padded_attention_mask, padded_labels, node_indices_tensor)` - all as tensors
5. **Checkpoint format**: `{epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict}`
6. **Labels padding value**: Always `-100` for ignored positions (PyTorch cross-entropy convention)

## Interfaces

### Upstream (from Root)
- Receives task decomposition and coordination directives

### Downstream (to L2 agents)
- **→ Preprocessing Agent**: Provides raw conversations, receives preprocessed text
- **→ Layers Orchestrator**: Provides SDR embeddings + adjacency matrix, receives model instance
- **→ Evaluation Agent**: Provides trained model + dataset, receives metric scores
- **→ Utilities Agent**: Provides config path, receives Config object + Dataset

### Key Function Signatures

```python
# main.py
load_dataset(config: Config) -> List[List[Dict[str, Any]]]
compute_and_save_adjacency_matrix(
    sdr_embeddings: torch.Tensor,
    threshold: float, batch_size: int,
    output_file: str, device: torch.device,
    fallback_mode: str, ...
) -> Tuple[Union[csr_matrix, torch.Tensor], int]

# train.py
train(model, dataloader, optimizer, scheduler, ...) -> metrics
collate_fn(batch) -> (input_ids, attention_mask, labels, node_indices)
save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir, model_name)
load_checkpoint(checkpoint_path, model, optimizer, scheduler)
```

## Delegation

- **Escalate to Root**: When a change affects multiple L2 agents simultaneously
- **Delegate to Preprocessing**: All text tokenization, vocabulary, and embedding changes
- **Delegate to Layers**: All neural architecture modifications
- **Delegate to Evaluation**: All metric computation and downstream task changes
- **Delegate to Utilities**: Config schema changes, dataset class modifications
- **Consult Testing**: After any pipeline change, verify test compatibility
