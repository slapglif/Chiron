# agents.md - Preprocessing Agent (L2)

## Role

Text preprocessing and embedding generation specialist. Transforms raw conversation data into dense vector representations suitable for downstream SDR generation. This agent owns the first two stages of the pipeline.

## Hermeneutic Context

Preprocessing is the **interpretive entry point** of the hermeneutic circle. Raw text is inherently ambiguous - this agent's role is to produce a first interpretation (tokenization, normalization, augmentation) that makes the text tractable for mathematical representation. The quality of all downstream representations depends entirely on decisions made here.

Understanding flows:
- **From the whole**: The pipeline needs consistent, normalized token sequences that Word2Vec can map to 300-dim vectors. This constrains preprocessing choices
- **To the whole**: Vocabulary size (max 1,000), frequency thresholds (min 5), and augmentation strategy shape what semantic information survives into SDRs

## Domain Expertise

- Text tokenization (whitespace-based splitting)
- Vocabulary construction with frequency filtering
- Text augmentation via WordNet synonym replacement (10% probability)
- OpenHermes-2.5 conversation format: `{from: "human"/"ai", value: "text"}`
- Word2Vec (Gensim KeyedVectors, GoogleNews-vectors-negative300.bin)
- Multithreaded batch processing (ThreadPoolExecutor)
- Cache integration (currently disabled but interface preserved)

## Owned Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `text_preprocessing.py` | Tokenization, vocab, augmentation | `TextPreprocessor` |
| `embedding.py` | Word2Vec embedding generation | `Word2VecEmbedding` |
| `__init__.py` | Package marker | - |

## Invariants

1. **Vocabulary always includes `<PAD>` (idx 0) and `<UNK>` (idx 1)** as the first two entries
2. **Preprocessing is deterministic** for the same input (except augmentation which uses `random`)
3. **Word2Vec vector size must be 300** - this is asserted against the loaded model
4. **Unknown tokens get zero vectors** (`np.zeros(300, dtype=np.float32)`)
5. **Text normalization**: lowercase, remove punctuation, replace digits with `<NUM>`
6. **Conversation format**: concatenate as `\nhuman:\n{human_text}\nai:\n{ai_text}\n`
7. **Batch validation**: each conversation turn must be a dict with `from` and `value` keys

## Interfaces

### Upstream (from Pipeline Agent)
```python
# Pipeline provides raw conversations
conversations: List[List[Dict[str, Any]]]
# Each conversation: [{"from": "human", "value": "..."}, {"from": "ai", "value": "..."}]
```

### Downstream (to SDR Agent via Pipeline)
```python
# TextPreprocessor.preprocess() output
preprocessed_texts: List[str]  # space-joined tokens

# Word2VecEmbedding.generate_embeddings() output
embeddings: List[np.ndarray]  # each shape (300,), dtype float32
```

### Lateral
- **→ Utilities Agent**: Uses `cache_data` / `load_cached_data` for caching (currently no-op)
- **→ SDR Agent**: Embedding dimensions (300) must match SDR input expectations

## Delegation

- **Escalate to Pipeline Agent**: If conversation format changes or new data sources are added
- **Consult SDR Agent**: If embedding dimensionality changes (affects projection dimensions)
- **Consult Utilities Agent**: If caching needs to be re-enabled

## Known Issues

- `text_preprocessing.py:207-228`: `_process_batch` method references `self.model` and `self.vector_size` which don't exist on `TextPreprocessor` - this is dead code copied from `embedding.py`
- Augmentation randomness means training is not fully reproducible without seeding `random`
