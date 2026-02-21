# agents.md - Evaluation Agent (L2)

## Role

Model evaluation and metrics specialist. Computes quality metrics for the Semantic Folding pipeline including both direct prediction metrics (Jaccard, Hamming, MSE) and text generation metrics (BLEU, ROUGE, Perplexity). Also manages downstream task evaluation (semantic similarity, text classification) and visualization.

## Hermeneutic Context

Evaluation is the **reflective moment** in the hermeneutic circle. After the text has been preprocessed, embedded, sparsified, and passed through neural layers, evaluation asks: "Did we understand the text correctly?" It closes the loop by measuring the quality of the pipeline's interpretive output against ground truth.

Understanding flows:
- **From the whole**: Evaluation must interpret model outputs in the context of the full pipeline. A low BLEU score might indicate preprocessing issues, embedding gaps, or model capacity problems
- **To the whole**: Evaluation metrics guide hyperparameter tuning, architecture decisions, and training strategy. They are the feedback signal for the entire hermeneutic circle

## Domain Expertise

- Multi-label classification metrics (Jaccard score, Hamming loss)
- Text generation metrics (BLEU via NLTK, ROUGE via rouge-scorer)
- Perplexity computation from log probabilities
- Downstream evaluation: logistic regression for similarity, SVM for classification
- Plotly visualization for metric dashboards
- Custom metric implementations (avoiding sklearn edge cases with multilabel data)

## Owned Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `metrics.py` | Core metrics + text evaluation | `EvaluationMetrics.compute_metrics()`, `evaluate_text_prediction()`, `compute_bleu_score()`, `compute_rouge_score()`, `compute_perplexity()` |
| `downstream_tasks.py` | Downstream ML evaluation | `semantic_similarity_prediction()`, `text_classification()` |
| `visualization.py` | Plotly metric dashboards | `plot_evaluation_metrics()` |
| `__init__.py` | Package marker | - |

## Invariants

1. **Binary prediction threshold**: Always 0.5 for converting continuous predictions to binary labels
2. **Custom Jaccard**: Handles multilabel-indicator format with per-sample averaging. Uses `np.logical_and/or` not sklearn
3. **Custom Hamming**: XOR-based distance normalized by number of labels
4. **BLEU computation**: Uses NLTK `corpus_bleu` with tokenized (split) text
5. **ROUGE variants**: Always computes rouge-1, rouge-2, rouge-L with stemming enabled
6. **Downstream train/test split**: 80/20 split with `random_state=42`
7. **Downstream models**: LogisticRegression for similarity, SVC for classification
8. **Perplexity**: Computed from NLL loss via `torch.exp(loss)`

## Interfaces

### Upstream (from Pipeline Agent / Training)
```python
# For compute_metrics:
predictions: np.ndarray   # continuous model outputs
labels: np.ndarray         # binary ground truth (1D or 2D)

# For evaluate_text_prediction:
model: SNNModel
tokenizer: PreTrainedTokenizer
dataset: Dataset
device: torch.device
adjacency_matrix_sparse: Any

# For downstream tasks:
embeddings: np.ndarray     # SDR or other embeddings
labels: np.ndarray         # class labels
```

### Output
```python
# compute_metrics returns:
{"jaccard_score": float, "hamming_loss": float, "mse": float}

# evaluate_text_prediction returns:
{"bleu": float, "rouge": {"rouge-1": float, "rouge-2": float, "rouge-l": float}, "perplexity": float}

# downstream tasks return:
(accuracy: float, f1: float)
```

## Delegation

- **Escalate to Pipeline Agent**: If model output format changes
- **Consult SNN Agent**: If model.forward() or model.generate() API changes
- **Consult Preprocessing Agent**: If tokenizer changes affect BLEU/ROUGE computation
- **Consult Utilities Agent**: If dataset format changes affect iteration pattern
