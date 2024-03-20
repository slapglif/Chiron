
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_evaluation_metrics(
    evaluation_scores: Dict[str, float],
    sem_sim_accuracy: float,
    sem_sim_f1: float,
    text_class_accuracy: float,
    text_class_f1: float,
) -> None:
    """
    Visualize the evaluation metrics using Plotly.

    Args:
        evaluation_scores (Dict[str, float]): Evaluation scores from the model.
        sem_sim_accuracy (float): Semantic similarity prediction accuracy.
        sem_sim_f1 (float): Semantic similarity prediction F1 score.
        text_class_accuracy (float): Text classification accuracy.
        text_class_f1 (float): Text classification F1 score.
    """
    metrics = ["Accuracy", "F1 Score", "MSE"]
    scores = [
        evaluation_scores["accuracy"],
        evaluation_scores["f1_score"],
        evaluation_scores["mse"],
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Model Evaluation Metrics",
            "Semantic Similarity Prediction",
            "Text Classification",
            "Downstream Tasks",
        ),
    )

    # Model Evaluation Metrics
    fig.add_trace(go.Bar(x=metrics, y=scores, name="Model Evaluation"), row=1, col=1)

    # Semantic Similarity Prediction
    fig.add_trace(
        go.Bar(
            x=["Accuracy", "F1 Score"],
            y=[sem_sim_accuracy, sem_sim_f1],
            name="Semantic Similarity Prediction",
        ),
        row=1,
        col=2,
    )

    # Text Classification
    fig.add_trace(
        go.Bar(
            x=["Accuracy", "F1 Score"],
            y=[text_class_accuracy, text_class_f1],
            name="Text Classification",
        ),
        row=2,
        col=1,
    )

    # Downstream Tasks
    downstream_metrics = [
        "Semantic Similarity Accuracy",
        "Semantic Similarity F1",
        "Text Classification Accuracy",
        "Text Classification F1",
    ]
    downstream_scores = [
        sem_sim_accuracy,
        sem_sim_f1,
        text_class_accuracy,
        text_class_f1,
    ]
    fig.add_trace(
        go.Bar(x=downstream_metrics, y=downstream_scores, name="Downstream Tasks"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Evaluation Metrics", showlegend=False, height=600, width=800
    )
    fig.show()