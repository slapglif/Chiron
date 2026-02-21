from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


def plot_evaluation_metrics(
    evaluation_scores: Dict[str, float],
    sem_sim_accuracy: float,
    sem_sim_f1: float,
    text_class_accuracy: float,
    text_class_f1: float,
) -> None:
    """
    Visualize the evaluation metrics using Plotly and log them as text.

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

    # Log the evaluation metrics as text
    logger.info("Evaluation Metrics:")
    logger.info("Model Evaluation Metrics:")
    for metric, score in zip(metrics, scores):
        logger.info(f"  {metric}: {score:.4f}")

    logger.info("\nSemantic Similarity Prediction:")
    logger.info(f"  Accuracy: {sem_sim_accuracy:.4f}")
    logger.info(f"  F1 Score: {sem_sim_f1:.4f}")

    logger.info("\nText Classification:")
    logger.info(f"  Accuracy: {text_class_accuracy:.4f}")
    logger.info(f"  F1 Score: {text_class_f1:.4f}")

    logger.info("\nDownstream Tasks:")
    for metric, score in zip(downstream_metrics, downstream_scores):
        logger.info(f"  {metric}: {score:.4f}")


# ---------------------------------------------------------------------------
# Training curve visualization
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot training (and optionally validation) loss and metrics over epochs.

    Args:
        train_losses (List[float]): Training loss values per epoch.
        val_losses (Optional[List[float]]): Validation loss values per epoch.
        train_metrics (Optional[Dict[str, List[float]]]): Additional training
            metrics keyed by name, each a list of values per epoch.
        val_metrics (Optional[Dict[str, List[float]]]): Additional validation
            metrics keyed by name, each a list of values per epoch.
        title (str): Overall figure title.
        save_path (Optional[str]): If provided, save the figure as an image.

    Returns:
        go.Figure: The Plotly figure object.
    """
    # Determine how many subplots we need: 1 for loss + 1 per metric
    metric_names = sorted(
        set(list((train_metrics or {}).keys()) + list((val_metrics or {}).keys()))
    )
    n_plots = 1 + len(metric_names)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    subplot_titles = ["Loss"] + [name for name in metric_names]
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    epochs = list(range(1, len(train_losses) + 1))

    # --- Loss subplot ---
    fig.add_trace(
        go.Scatter(
            x=epochs, y=train_losses, mode="lines+markers",
            name="Train Loss", line=dict(color="royalblue"),
        ),
        row=1, col=1,
    )
    if val_losses is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_losses) + 1)), y=val_losses,
                mode="lines+markers", name="Val Loss",
                line=dict(color="tomato", dash="dash"),
            ),
            row=1, col=1,
        )

    # --- Metric subplots ---
    colors_train = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    colors_val = ["#ff7f0e", "#d62728", "#bcbd22", "#17becf", "#7f7f7f"]

    for idx, name in enumerate(metric_names):
        row = (idx + 1) // n_cols + 1
        col = (idx + 1) % n_cols + 1

        if train_metrics and name in train_metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(train_metrics[name]) + 1)),
                    y=train_metrics[name],
                    mode="lines+markers",
                    name=f"Train {name}",
                    line=dict(color=colors_train[idx % len(colors_train)]),
                ),
                row=row, col=col,
            )

        if val_metrics and name in val_metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(val_metrics[name]) + 1)),
                    y=val_metrics[name],
                    mode="lines+markers",
                    name=f"Val {name}",
                    line=dict(
                        color=colors_val[idx % len(colors_val)], dash="dash"
                    ),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=title, height=350 * n_rows, width=400 * n_cols, showlegend=True,
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Value")

    if save_path:
        fig.write_image(save_path)
        logger.info(f"Training curves saved to {save_path}")

    fig.show()
    return fig


# ---------------------------------------------------------------------------
# Embedding space visualization (t-SNE / UMAP)
# ---------------------------------------------------------------------------

def plot_embedding_space(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    title: str = "Embedding Space",
    save_path: Optional[str] = None,
    sample_size: Optional[int] = 5000,
    random_state: int = 42,
) -> go.Figure:
    """
    Project high-dimensional embeddings into 2-D using t-SNE or UMAP and
    create a scatter plot, optionally colored by labels.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (n_samples, embed_dim).
        labels (Optional[np.ndarray]): Per-sample labels for coloring.
        method (str): Projection method, one of 'tsne' or 'umap'. Defaults to 'tsne'.
        perplexity (float): Perplexity for t-SNE. Defaults to 30.0.
        n_neighbors (int): Number of neighbors for UMAP. Defaults to 15.
        title (str): Figure title.
        save_path (Optional[str]): If provided, save the figure as an image.
        sample_size (Optional[int]): If not None, randomly subsample to this
            many points before projection (for performance). Defaults to 5000.
        random_state (int): Random seed.

    Returns:
        go.Figure: The Plotly figure object.
    """
    rng = np.random.RandomState(random_state)

    # Subsample for large datasets
    if sample_size is not None and embeddings.shape[0] > sample_size:
        indices = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[indices]
        if labels is not None:
            labels = np.asarray(labels)[indices]
        logger.info(f"Subsampled to {sample_size} points for embedding visualization.")

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=random_state, init="pca", learning_rate="auto",
        )
        coords = reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            logger.error(
                "umap-learn is required for UMAP projection. "
                "Install it with: pip install umap-learn"
            )
            raise
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported projection method: {method}. Use 'tsne' or 'umap'.")

    fig = go.Figure()

    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1],
                    mode="markers", name=str(lbl),
                    marker=dict(size=4, opacity=0.7),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode="markers", name="embeddings",
                marker=dict(size=4, opacity=0.7, color="royalblue"),
            )
        )

    fig.update_layout(
        title=f"{title} ({method.upper()})",
        xaxis_title=f"{method.upper()} Dim 1",
        yaxis_title=f"{method.upper()} Dim 2",
        height=600, width=700, showlegend=True,
    )

    if save_path:
        fig.write_image(save_path)
        logger.info(f"Embedding space plot saved to {save_path}")

    fig.show()
    return fig


# ---------------------------------------------------------------------------
# Attention weight visualization
# ---------------------------------------------------------------------------

def plot_attention_weights(
    attention_weights: np.ndarray,
    head_index: int = 0,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Visualize attention weights as a heatmap.

    Supports single-head or multi-head attention tensors.

    Args:
        attention_weights (np.ndarray): Attention weight matrix.  Accepted
            shapes:
            - (seq_len, seq_len): single attention matrix.
            - (n_heads, seq_len, seq_len): multi-head; *head_index* selects
              which head to display.
            - (batch, n_heads, seq_len, seq_len): batch of multi-head;
              first sample and *head_index* are used.
        head_index (int): Which attention head to visualize when the tensor
            has a head dimension. Defaults to 0.
        tokens (Optional[List[str]]): Token strings for axis labels. If None,
            integer indices are used.
        title (str): Figure title.
        save_path (Optional[str]): If provided, save the figure as an image.

    Returns:
        go.Figure: The Plotly figure object.
    """
    attn = np.asarray(attention_weights)

    # Collapse to 2-D (seq_len, seq_len)
    if attn.ndim == 4:
        attn = attn[0, head_index]
    elif attn.ndim == 3:
        attn = attn[head_index]
    elif attn.ndim != 2:
        raise ValueError(
            f"Unsupported attention_weights shape: {attention_weights.shape}. "
            "Expected 2-D, 3-D, or 4-D."
        )

    seq_len = attn.shape[0]
    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]

    fig = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            colorbar=dict(title="Weight"),
        )
    )

    fig.update_layout(
        title=f"{title} (Head {head_index})",
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        height=600, width=650,
        yaxis=dict(autorange="reversed"),
    )

    if save_path:
        fig.write_image(save_path)
        logger.info(f"Attention weight plot saved to {save_path}")

    fig.show()
    return fig


# ---------------------------------------------------------------------------
# SDR sparsity distribution visualization
# ---------------------------------------------------------------------------

def plot_sdr_sparsity_distribution(
    sdr_matrix: np.ndarray,
    title: str = "SDR Sparsity Distribution",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Visualize the sparsity distribution of Sparse Distributed Representations.

    Creates a two-panel figure:
    1. Histogram of per-sample sparsity (fraction of active bits).
    2. Bar chart showing activation frequency for each bit position.

    Args:
        sdr_matrix (np.ndarray): Binary SDR matrix of shape (n_samples, sdr_dim).
            Values should be 0 or 1 (or float, thresholded at 0.5).
        title (str): Overall figure title.
        save_path (Optional[str]): If provided, save the figure as an image.

    Returns:
        go.Figure: The Plotly figure object.
    """
    sdr = np.asarray(sdr_matrix)
    if sdr.ndim == 1:
        sdr = sdr[np.newaxis, :]

    # Binarize if not already binary
    binary_sdr = (sdr > 0.5).astype(float)

    n_samples, sdr_dim = binary_sdr.shape

    # Per-sample sparsity: fraction of active bits
    sparsities = binary_sdr.mean(axis=1)

    # Per-bit activation frequency across all samples
    bit_frequencies = binary_sdr.mean(axis=0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Per-Sample Sparsity", "Per-Bit Activation Frequency"],
        horizontal_spacing=0.1,
    )

    # Panel 1: Sparsity histogram
    fig.add_trace(
        go.Histogram(
            x=sparsities, nbinsx=50,
            marker_color="royalblue", name="Sparsity",
            hovertemplate="Sparsity: %{x:.3f}<br>Count: %{y}",
        ),
        row=1, col=1,
    )

    # Panel 2: Bit activation frequency
    # If sdr_dim is very large, subsample bit positions for readability
    max_bars = 200
    if sdr_dim > max_bars:
        step = sdr_dim // max_bars
        bit_indices = list(range(0, sdr_dim, step))
        bit_freq_display = bit_frequencies[::step]
    else:
        bit_indices = list(range(sdr_dim))
        bit_freq_display = bit_frequencies

    fig.add_trace(
        go.Bar(
            x=bit_indices, y=bit_freq_display,
            marker_color="tomato", name="Bit Freq",
            hovertemplate="Bit %{x}<br>Freq: %{y:.4f}",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Sparsity (fraction active)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Bit Position", row=1, col=2)
    fig.update_yaxes(title_text="Activation Frequency", row=1, col=2)

    fig.update_layout(
        title=title, height=450, width=1000, showlegend=False,
    )

    # Log summary statistics
    logger.info(f"SDR Sparsity Distribution ({n_samples} samples, {sdr_dim} dims):")
    logger.info(f"  Mean sparsity: {np.mean(sparsities):.4f}")
    logger.info(f"  Std sparsity:  {np.std(sparsities):.4f}")
    logger.info(f"  Min sparsity:  {np.min(sparsities):.4f}")
    logger.info(f"  Max sparsity:  {np.max(sparsities):.4f}")
    logger.info(f"  Mean bit freq: {np.mean(bit_frequencies):.4f}")

    if save_path:
        fig.write_image(save_path)
        logger.info(f"SDR sparsity plot saved to {save_path}")

    fig.show()
    return fig
