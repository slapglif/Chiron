# agents.md - Utilities Agent (L2)

## Role

Infrastructure and shared services specialist. Manages configuration loading, dataset construction, and caching utilities. This agent provides the foundational services that all other agents depend on.

## Hermeneutic Context

Utilities represent the **structural pre-conditions** for the hermeneutic circle. Before any interpretation can begin, the system needs configuration (what parameters shape our interpretation?), data access (what are we interpreting?), and caching (how do we remember past interpretations?). These are the conditions of possibility for understanding.

Understanding flows:
- **From the whole**: Config schema must reflect all hyperparameters needed by every pipeline stage. Dataset class must serve data in the format the training loop expects
- **To the whole**: Config changes propagate to every consumer. Dataset format changes affect DataLoader, collate_fn, and training loop

## Domain Expertise

- JSON configuration management
- PyTorch Dataset/DataLoader patterns
- HuggingFace Tokenizer integration (BERT tokenizer for SDR-to-text encoding)
- Pandas/NumPy data serialization (Feather format, currently disabled)
- Cache invalidation strategies

## Owned Files

| File | Class | Purpose |
|------|-------|---------|
| `config.py` | `Config` | JSON config loader with dict-like access (`__getitem__`, `get`, `__contains__`, `update`, `save`) |
| `data.py` | `SemanticFoldingDataset` | PyTorch Dataset wrapping SDR embeddings + BERT tokenizer |
| `cache.py` | - | `cache_data()` and `load_cached_data()` (currently disabled - early return) |
| `__init__.py` | - | Package marker |

## Invariants

### Config
1. **Immutable at runtime**: Config is loaded once from JSON and should not be modified during pipeline execution (except via explicit `update()`)
2. **Required top-level keys**: `dataset_params`, `longformer_params`, `max_sequence_length`, `preprocessing_params`, `embedding_params`, `sdr_params`, `htm_params`, `snn_params`, `gat_params`, `device`, `batch_size`, `num_epochs`, `learning_rate`, `adjacency_matrix`, `accumulation_steps`, `checkpoint_dir`, `num_workers`, `k_folds`, `patience`, `tokenizer`, `neptune_key`
3. **No validation**: Config class does not validate keys or types - consumers must handle missing/invalid values

### SemanticFoldingDataset
4. **Returns 4-tuple**: `(input_ids, attention_mask, label, node_index)`
5. **SDR-to-text encoding**: SDR embeddings are converted to space-separated string representation, then BERT-tokenized. This is unconventional but intentional
6. **Default labels**: If no labels provided, creates zero tensors of shape `(N, max_seq_len)`
7. **Label padding**: Pads with `-100` (PyTorch ignore index) or truncates to `max_seq_len`
8. **Attention mask**: Always all-ones (no padding detection)

### Cache
9. **Currently disabled**: Both functions have `return` as the first statement - all code after is unreachable
10. **Interface preserved**: Other modules import and call cache functions, which silently no-op

## Interfaces

### Config consumers (all agents)
```python
config = Config("chiron/config.json")
config["dataset_params"]           # dict access
config.get("missing_key", default) # safe access
"key" in config                    # containment check
```

### Dataset consumers (Pipeline Agent, Training)
```python
dataset = SemanticFoldingDataset(
    sdr_embeddings=torch.Tensor,  # (N, sdr_dim)
    tokenizer=PreTrainedTokenizer,  # BERT tokenizer
    labels=torch.Tensor,  # optional, (N, seq_len)
    max_seq_len=int  # default 10
)
# Iteration yields:
(input_ids: Tensor, attention_mask: Tensor, label: Tensor, node_index: int)
```

### Cache consumers (Preprocessing Agent)
```python
cached = load_cached_data("cache_key")  # always returns None
cache_data(data, "cache_key")           # no-op
```

## Delegation

- **Escalate to Pipeline Agent**: If config schema changes
- **Broadcast to all agents**: When new config keys are added (all consumers may need updates)
- **Consult Preprocessing Agent**: If caching needs to be re-enabled
