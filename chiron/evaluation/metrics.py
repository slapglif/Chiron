from typing import Any, Dict, List

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from numpy import ndarray
from rouge_score import rouge_scorer
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

        # Compute mean squared error
        squared_errors = []
        for i in range(binary_labels.shape[0]):
            squared_errors.append(
                np.mean((binary_labels[i].astype(float) - predictions[i]) ** 2)
            )
        mse = np.mean(squared_errors)

        metrics = {
            "jaccard_score": jaccard,
            "hamming_loss": hamming,
            "mse": mse,
        }
        return metrics


def custom_jaccard_score(y_true, y_pred, average="binary"):
    """
    Custom implementation of the Jaccard score for multilabel-indicator targets.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        average (str, optional): Averaging method. Defaults to 'binary'.

    Returns:
        float: Jaccard score.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    intersection = np.sum(np.logical_and(y_true, y_pred), axis=1)
    union = np.sum(np.logical_or(y_true, y_pred), axis=1)

    jaccard_scores = intersection / union

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


def evaluate_text_prediction(
    model: SNNModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: torch.device,
    max_length: int = 1024,
    num_return_sequences: int = 1,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Evaluate the text prediction performance of a model.

    Args:
        model (SNNModel): The semantic folding model.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        dataset (torch.utils.data.Dataset): The dataset to evaluate on.
        device (torch.device): The device to run the evaluation on.
        max_length (int): The maximum length of the generated text.
        num_return_sequences (int): The number of sequences to generate.
        **kwargs (Any): Additional keyword arguments for text generation.

    Returns:
        Dict[str, float]: The evaluation metrics.
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

            # Generate text
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )

            # Decode the generated text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

            # Decode the reference text
            reference_text = tokenizer.decode(labels, skip_special_tokens=True)
            reference_texts.append(reference_text)

    # Compute evaluation metrics
    metrics["bleu"] = compute_bleu_score(generated_texts, reference_texts)
    metrics["rouge"] = compute_rouge_score(generated_texts, reference_texts)
    metrics["perplexity"] = compute_perplexity(
        generated_texts, model, tokenizer, device
    )

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
