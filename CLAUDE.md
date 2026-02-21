# CLAUDE.md - Chiron Project Intelligence

## Project Identity

Chiron is a **Semantic Folding Neural Network Pipeline** that combines biologically-inspired algorithms (HTM, SNN, SDR) with modern deep learning (GAT, Transformers) to generate and process sparse distributed semantic representations of text.

## Architecture Overview

```
Text Data (OpenHermes-2.5)
    |
    v
TextPreprocessor (tokenize, augment, vocabulary)
    |
    v
Word2VecEmbedding (GoogleNews 300-dim pretrained)
    |
    v
SDRGenerator (TruncatedSVD 64-dim projection -> 256-dim binary SDR)
    |
    v
Adjacency Matrix (cosine similarity -> threshold -> sparse CSR)
    |
    v
SNNModel (orchestrates SNN + GAT + HTM layers)
    |-- SNNLayer: spiking neurons, 10 timesteps, membrane potential
    |-- GraphAttentionLayer: multi-head attention over adjacency graph
    |-- HTMModel: spatial pooler with minicolumns, boosting, permanence
    |-- fc_out: final linear projection to SDR dimensions
    |
    v
Evaluation (BLEU, ROUGE, Perplexity, Jaccard, Hamming, downstream tasks)
```

## Tech Stack

- **Language**: Python 3.x
- **Core ML**: PyTorch, Transformers (HuggingFace), NumPy, SciPy, scikit-learn
- **NLP**: Gensim (Word2Vec), NLTK, SpaCy, sentence-transformers
- **Data**: HuggingFace Datasets, Pandas
- **Tokenizer**: BERT (bert-base-uncased)
- **Monitoring**: Neptune, TensorBoard, Loguru
- **Visualization**: Plotly, Matplotlib
- **GPU**: CUDA, Accelerate, Bitsandbytes

## Critical Paths

| Entry Point | Purpose |
|---|---|
| `chiron/main.py` | Full pipeline orchestration - dataset loading through training |
| `chiron/train.py` | Training loop with checkpointing, gradient accumulation, early stopping |
| `chiron/pipeline.py` | Text prediction inference pipeline |
| `chiron/config.json` | All hyperparameters and model configuration |

## Key Conventions

### Code Style
- Docstrings on all public methods (Google-style with Args/Returns)
- Type hints throughout (typing module)
- Loguru for structured logging (`logger.info`, `logger.debug`, `logger.warning`)
- tqdm progress bars for long-running operations

### Configuration
- All hyperparameters live in `chiron/config.json` - never hardcode
- Config loaded via `chiron/utils/config.py` Config class (dict-like access)
- Device configuration: `cuda:0` by default

### Data Flow Invariants
- SDR embeddings are binary (0/1) after thresholding
- SDR sparsity target: 0.2% active bits
- Word2Vec embeddings: 300-dimensional float vectors
- Adjacency matrices stored as scipy.sparse.csr_matrix
- Conversations from OpenHermes-2.5 have `{from: "human"/"ai", value: "..."}` format

### Training Configuration
- Batch size: 2,056
- Epochs: 20
- Learning rate: 0.0001
- Gradient accumulation: 16 steps
- K-fold cross-validation: 5 folds
- Early stopping patience: 5
- Checkpoint dir: `.checkpoints/`

### Tensor Shape Conventions
- Input sequences: `(batch_size, seq_len, input_size)`
- SNN output: `(batch_size, seq_len, timesteps, output_size)`
- GAT expects: `(batch_size, seq_len, num_features)` with `(seq_len, seq_len)` adjacency
- HTM operates on numpy internally, converts to/from torch at boundaries

## Module Dependency Graph

```
main.py
├── preprocessing/text_preprocessing.py  (TextPreprocessor)
├── preprocessing/embedding.py           (Word2VecEmbedding)
├── layers/sdr/sdr_generation.py         (SDRGenerator)
├── layers/snn/model.py                  (SNNModel, SNNLayer)
│   ├── layers/snn/graph_attention.py    (GraphAttentionLayer)
│   └── layers/htm/model.py             (HTMModel, HTMSpatialPooler)
├── train.py                             (train, collate_fn)
├── utils/config.py                      (Config)
├── utils/data.py                        (SemanticFoldingDataset)
└── evaluation/metrics.py                (EvaluationMetrics)
```

## Known Quirks

- `utils/cache.py`: Caching is **disabled** - both `cache_data` and `load_cached_data` return immediately (early `return` before logic)
- `text_preprocessing.py` has a `_process_batch` method that references `self.model` which doesn't exist on TextPreprocessor - this is dead/copied code from embedding.py
- HTM spatial pooler converts torch tensors to numpy for computation and back - this breaks gradient flow intentionally (HTM is non-differentiable)
- Neptune API key is base64-encoded in config.json
- `SNNModel.forward` stores visualization data on every forward pass in `self.visualization_data`

## Build & Run

```bash
cd chiron
pip install -r requirements.txt
python -m chiron.main --config chiron/config.json
```

## Testing

```bash
pytest chiron/tests/
```

Tests currently cover HTM spatial pooler only (`chiron/tests/htm/pooler/test_HTMSpatialPooler.py`).
