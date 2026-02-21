from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from nltk.translate.bleu_score import corpus_bleu
from numpy import ndarray
from rouge_score import rouge_scorer
from scipy.stats import spearmanr
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chiron.layers.snn.model import SNNModel


class EvaluationMetrics:
    """Class for computing evaluation metrics."""

    @staticmethod
    def compute_metrics(
        predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics given the predicted and true labels.

        Args:
            predictions (np.ndarray): Array of predicted labels.
            labels (np.ndarray): Array of true labels.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        # Convert predictions to binary labels using a threshold
        threshold = 0.5
        binary_predictions = (predictions > threshold).astype(int)

        # Handle different shapes of labels array
        if labels.ndim == 1:
            # Labels are already in binary format
            binary_labels = labels.astype(int)
        elif labels.ndim == 2:
            # Labels are in multilabel-indicator format
            binary_labels = labels
            if binary_labels.shape[0] != binary_predictions.shape[0]:
                # Ensure that binary_predictions and binary_labels have the same number of samples
                binary_predictions = binary_predictions[: binary_labels.shape[0]]
        else:
            raise ValueError(f"Unsupported labels shape: {labels.shape}")

        # Compute custom Jaccard score for multilabel-indicator targets
        jaccard = custom_jaccard_score(
            binary_labels, binary_predictions, average="samples"
        )

        # Compute custom Hamming loss for multilabel-indicator targets
        hamming = custom_hamming_loss(binary_labels, binary_predictions)

        # Compute mean squared error (vectorized)
        mse = np.mean((binary_labels.astype(float) - predictions) ** 2)

        metrics = {
            "jaccard_score": jaccard,
            "hamming_loss": hamming,
            "mse": mse,
        }
        return metrics


def custom_jaccard_score(y_true, y_pred, average="binary", eps=1e-8):
    """
    Custom implementation of the Jaccard score for multilabel-indicator targets.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        average (str, optional): Averaging method. Defaults to 'binary'.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-8.

    Returns:
        float: Jaccard score.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    intersection = np.sum(np.logical_and(y_true, y_pred), axis=1)
    union = np.sum(np.logical_or(y_true, y_pred), axis=1)

    jaccard_scores = intersection / (union + eps)

    if average == "binary":
        return np.mean(jaccard_scores)
    elif average == "samples":
        return np.mean(jaccard_scores)
    else:
        raise ValueError(f"Invalid value for 'average': {average}")


def custom_hamming_loss(y_true, y_pred):
    """
    Custom implementation of the Hamming loss for multilabel-indicator targets.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        float: Hamming loss.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    hamming_distance = np.sum(np.logical_xor(y_true, y_pred), axis=1)
    hamming_loss = hamming_distance / y_true.shape[1]

    return np.mean(hamming_loss)


# ---------------------------------------------------------------------------
# Embedding quality metrics (Wang & Isola, 2020)
# ---------------------------------------------------------------------------

def compute_alignment(
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    alpha: float = 2.0,
) -> float:
    """
    Compute the alignment metric (Wang & Isola, 2020).

    Alignment measures how close embeddings of similar (positive) pairs are.
    Lower values indicate better alignment.

    alignment = E[ ||f(x) - f(y)||^alpha ]  for positive pairs (x, y)

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (n_samples, embed_dim).
        positive_pairs (List[Tuple[int, int]]): List of index pairs that are
            considered semantically similar / positive.
        alpha (float): Exponent for the distance. Defaults to 2.0.

    Returns:
        float: The alignment score.
    """
    if len(positive_pairs) == 0:
        logger.warning("No positive pairs provided for alignment computation.")
        return 0.0

    pair_indices = np.array(positive_pairs)
    diffs = embeddings[pair_indices[:, 0]] - embeddings[pair_indices[:, 1]]
    distances = np.linalg.norm(diffs, axis=1) ** alpha
    return float(np.mean(distances))


def compute_uniformity(embeddings: np.ndarray, t: float = 2.0) -> float:
    """
    Compute the uniformity metric (Wang & Isola, 2020).

    Uniformity measures how uniformly the embeddings are distributed on the
    unit hypersphere.  Lower (more negative) values indicate better uniformity.

    uniformity = log E[ e^{-t * ||f(x) - f(y)||^2} ]

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (n_samples, embed_dim).
            Embeddings are L2-normalized internally.
        t (float): Temperature parameter. Defaults to 2.0.

    Returns:
        float: The uniformity score.
    """
    # L2-normalize embeddings onto the unit hypersphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms

    n = normed.shape[0]
    if n < 2:
        logger.warning("Need at least 2 embeddings for uniformity computation.")
        return 0.0

    # Compute pairwise squared distances using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (all norms are 1)
    dot_products = normed @ normed.T
    sq_distances = 2.0 - 2.0 * dot_products

    # Extract upper-triangle (unique pairs, excluding self-pairs)
    triu_indices = np.triu_indices(n, k=1)
    sq_dists_pairs = sq_distances[triu_indices]

    # log E[exp(-t * ||f(x)-f(y)||^2)]
    uniformity = np.log(np.mean(np.exp(-t * sq_dists_pairs)))
    return float(uniformity)


# ---------------------------------------------------------------------------
# Representation Similarity Analysis (RSA)
# ---------------------------------------------------------------------------

def compute_rsa(
    input_features: np.ndarray,
    representations: np.ndarray,
    metric: str = "correlation",
) -> float:
    """
    Compute Representation Similarity Analysis (RSA).

    Measures the correlation between an input-space similarity matrix (computed
    from *input_features*) and a representation-space similarity matrix
    (computed from *representations*).

    Args:
        input_features (np.ndarray): Original input feature matrix (n_samples, input_dim).
        representations (np.ndarray): Learned representation matrix (n_samples, repr_dim).
        metric (str): Similarity metric. Currently supports 'correlation' (Pearson)
            and 'cosine'. Defaults to 'correlation'.

    Returns:
        float: Spearman rank correlation between the two similarity matrices.
    """
    def _similarity_matrix(X: np.ndarray, metric: str) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        if metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            X_normed = X / norms
            return X_normed @ X_normed.T
        elif metric == "correlation":
            X_centered = X - X.mean(axis=1, keepdims=True)
            stds = np.linalg.norm(X_centered, axis=1, keepdims=True)
            stds = np.maximum(stds, 1e-8)
            X_normed = X_centered / stds
            return X_normed @ X_normed.T
        else:
            raise ValueError(f"Unsupported RSA metric: {metric}")

    sim_input = _similarity_matrix(input_features, metric)
    sim_repr = _similarity_matrix(representations, metric)

    # Use upper-triangle entries (exclude diagonal self-similarities)
    n = sim_input.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vec_input = sim_input[triu_idx]
    vec_repr = sim_repr[triu_idx]

    correlation, _ = spearmanr(vec_input, vec_repr)
    return float(correlation)


# ---------------------------------------------------------------------------
# BERTScore using sentence-transformers
# ---------------------------------------------------------------------------

def compute_bert_score(
    generated_texts: List[str],
    reference_texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute BERTScore-style metric using sentence-transformers.

    Encodes both generated and reference texts into embeddings and computes
    cosine similarity as a proxy for semantic equivalence.

    Args:
        generated_texts (List[str]): Generated/hypothesis texts.
        reference_texts (List[str]): Reference/ground-truth texts.
        model_name (str): Sentence-transformer model name. Defaults to 'all-MiniLM-L6-v2'.
        device (Optional[str]): Device string ('cpu', 'cuda'). If None, auto-detected.

    Returns:
        Dict[str, float]: Dictionary with 'bert_score_mean' and 'bert_score_std'.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers is required for BERTScore. "
            "Install it with: pip install sentence-transformers"
        )
        return {"bert_score_mean": 0.0, "bert_score_std": 0.0}

    st_model = SentenceTransformer(model_name, device=device)

    gen_embeddings = st_model.encode(generated_texts, convert_to_numpy=True)
    ref_embeddings = st_model.encode(reference_texts, convert_to_numpy=True)

    # Row-wise cosine similarity
    gen_norms = np.linalg.norm(gen_embeddings, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    gen_norms = np.maximum(gen_norms, 1e-8)
    ref_norms = np.maximum(ref_norms, 1e-8)

    cosine_similarities = np.sum(
        (gen_embeddings / gen_norms) * (ref_embeddings / ref_norms), axis=1
    )

    return {
        "bert_score_mean": float(np.mean(cosine_similarities)),
        "bert_score_std": float(np.std(cosine_similarities)),
    }


# ---------------------------------------------------------------------------
# Text prediction evaluation (fixed to use forward pass, not generate())
# ---------------------------------------------------------------------------

def evaluate_text_prediction(
    model: SNNModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: torch.device,
    max_length: int = 1024,
    num_return_sequences: int = 1,
    adjacency_matrix_sparse: Any = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Evaluate text prediction quality using the model's forward pass.

    Uses the model forward pass to obtain logits and then greedily decodes
    predictions.  This avoids calling model.generate() which does not exist
    on SNNModel.

    Args:
        model (SNNModel): The SNN model.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding.
        dataset (Dataset): Evaluation dataset yielding
            (input_ids, attention_mask, labels, node_indices) tuples.
        device (torch.device): Device for computation.
        max_length (int): Maximum sequence length (unused, kept for API compat).
        num_return_sequences (int): Unused, kept for API compatibility.
        adjacency_matrix_sparse: Adjacency matrix for the model forward pass.
        **kwargs: Additional keyword arguments (kept for API compatibility).

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics (BLEU, ROUGE,
            perplexity, BERTScore).
    """
    model.eval()
    metrics = {}

    generated_texts = []
    reference_texts = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, node_indices in dataset:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)

            # Forward pass to obtain logits
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                adjacency_matrix=adjacency_matrix_sparse,
                node_indices=node_indices,
            )

            # Greedy decode: take argmax over the output dimension
            logger.debug(f"logits shape: {logits.shape}")
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the predicted token ids into text
            generated_text = tokenizer.decode(
                predicted_ids.view(-1).cpu(), skip_special_tokens=True
            )
            generated_texts.append(generated_text)

            # Decode the reference text
            reference_text = tokenizer.decode(
                labels.view(-1).cpu(), skip_special_tokens=True
            )
            reference_texts.append(reference_text)

    # Compute evaluation metrics
    metrics["bleu"] = compute_bleu_score(generated_texts, reference_texts)
    metrics["rouge"] = compute_rouge_score(generated_texts, reference_texts)
    metrics["perplexity"] = compute_perplexity(
        generated_texts, model, tokenizer, device
    )
    metrics["bert_score"] = compute_bert_score(generated_texts, reference_texts)

    return metrics


def compute_bleu_score(generated_texts: List[str], reference_texts: List[str]) -> float:
    """
    Compute the BLEU score between generated texts and reference texts.

    Args:
        generated_texts (List[str]): The list of generated texts.
        reference_texts (List[str]): The list of reference texts.

    Returns:
        float: The BLEU score.
    """
    # Tokenize the reference texts
    reference_texts = [[text.split()] for text in reference_texts]

    # Tokenize the generated texts
    generated_texts = [text.split() for text in generated_texts]

    # Compute the BLEU score
    bleu_score = corpus_bleu(reference_texts, generated_texts)

    return bleu_score


def compute_rouge_score(
    generated_texts: List[str], reference_texts: List[str]
) -> Dict[str, float]:
    """
    Compute the ROUGE score between generated texts and reference texts.

    Args:
        generated_texts (List[str]): The list of generated texts.
        reference_texts (List[str]): The list of reference texts.

    Returns:
        Dict[str, float]: The ROUGE scores (rouge-1, rouge-2, rouge-l).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_scores = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
    }

    for generated_text, reference_text in zip(generated_texts, reference_texts):
        scores = scorer.score(reference_text, generated_text)
        rouge_scores["rouge-1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge-2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rouge-l"].append(scores["rougeL"].fmeasure)

    avg_rouge_scores = {
        metric: np.mean(scores) for metric, scores in rouge_scores.items()
    }

    return avg_rouge_scores


def compute_perplexity(
    generated_texts: List[str],
    model: SNNModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> ndarray:
    """
    Compute the perplexity of the generated texts.

    Args:
        generated_texts (List[str]): The list of generated texts.
        model (SNNModel): The semantic folding model.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        device (torch.device): The device to run the computation on.

    Returns:
        float: The perplexity score.
    """
    perplexities = []

    for text in generated_texts:
        input_ids = torch.tensor(tokenizer.encode(text)).to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=None, adjacency_matrix=None)
            logits = outputs.view(-1, outputs.size(-1))
            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = input_ids.view(-1)
            loss = torch.nn.functional.nll_loss(log_probs, target_ids, reduction="mean")
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

    avg_perplexity = np.mean(perplexities)

    return avg_perplexity
