
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score


class EvaluationMetrics:
    """Class for computing evaluation metrics."""

    @staticmethod
    def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """
        Compute evaluation metrics given the predicted and true labels.

        Args:
            predictions (List[int]): List of predicted labels.
            labels (List[int]): List of true labels.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")

        metrics = {"accuracy": accuracy, "f1_score": f1}
        return metrics
