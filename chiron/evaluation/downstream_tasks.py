from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC


def semantic_similarity_prediction(
    embeddings,
    labels,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Evaluate embeddings on a semantic similarity prediction task using
    logistic regression with stratified k-fold cross-validation.

    Args:
        embeddings: Array-like of shape (n_samples, embed_dim).
        labels: Array-like of shape (n_samples,).
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[float, float, float, float]: Mean accuracy, F1, precision, recall
            across all folds.
    """
    # Split data into training and testing sets (kept for backward compat)
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=random_state
    )

    # Train a logistic regression model with cross-validation on the train split
    model = LogisticRegression(max_iter=1000, random_state=random_state)

    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        model, train_embeddings, train_labels, cv=cv, scoring=scoring,
        return_train_score=False,
    )

    logger.info(
        f"Semantic similarity CV results (n_folds={n_folds}): "
        f"accuracy={np.mean(cv_results['test_accuracy']):.4f} +/- {np.std(cv_results['test_accuracy']):.4f}, "
        f"f1={np.mean(cv_results['test_f1_weighted']):.4f} +/- {np.std(cv_results['test_f1_weighted']):.4f}"
    )

    # Refit on full training set and evaluate on held-out test set
    model.fit(train_embeddings, train_labels)
    predictions = model.predict(test_embeddings)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")
    precision = precision_score(test_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(test_labels, predictions, average="weighted", zero_division=0)

    return accuracy, f1, precision, recall


def text_classification(
    embeddings,
    labels,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Evaluate embeddings on a text classification task using an SVM
    classifier with stratified k-fold cross-validation.

    Args:
        embeddings: Array-like of shape (n_samples, embed_dim).
        labels: Array-like of shape (n_samples,).
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[float, float, float, float]: Mean accuracy, F1, precision, recall
            on the held-out test set.
    """
    # Split data into training and testing sets
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=random_state
    )

    # Train an SVM classifier with cross-validation on the train split
    classifier = SVC(kernel="rbf", random_state=random_state)

    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        classifier, train_embeddings, train_labels, cv=cv, scoring=scoring,
        return_train_score=False,
    )

    logger.info(
        f"Text classification CV results (n_folds={n_folds}): "
        f"accuracy={np.mean(cv_results['test_accuracy']):.4f} +/- {np.std(cv_results['test_accuracy']):.4f}, "
        f"f1={np.mean(cv_results['test_f1_weighted']):.4f} +/- {np.std(cv_results['test_f1_weighted']):.4f}"
    )

    # Refit on full training set and evaluate on held-out test set
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")
    precision = precision_score(test_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(test_labels, predictions, average="weighted", zero_division=0)

    return accuracy, f1, precision, recall


# ---------------------------------------------------------------------------
# Clustering quality evaluation
# ---------------------------------------------------------------------------

def clustering_evaluation(
    embeddings,
    labels,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate embedding quality via KMeans clustering quality metrics.

    Clusters the embeddings using KMeans and compares the resulting cluster
    assignments against the ground-truth labels using Normalized Mutual
    Information (NMI) and Adjusted Rand Index (ARI).

    Args:
        embeddings: Array-like of shape (n_samples, embed_dim).
        labels: Array-like of shape (n_samples,) with ground-truth class labels.
        n_clusters (Optional[int]): Number of clusters. If None, inferred from
            the number of unique labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        Dict[str, float]: Dictionary with keys 'nmi' and 'ari'.
    """
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)

    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)

    logger.info(f"Clustering evaluation: NMI={nmi:.4f}, ARI={ari:.4f}")

    return {"nmi": float(nmi), "ari": float(ari)}


# ---------------------------------------------------------------------------
# Retrieval task evaluation
# ---------------------------------------------------------------------------

def retrieval_evaluation(
    query_embeddings,
    corpus_embeddings,
    query_labels,
    corpus_labels,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate embedding quality via a retrieval task.

    For each query, ranks all corpus items by cosine similarity and computes
    Recall@K and Mean Reciprocal Rank (MRR).

    A retrieval is considered relevant if the corpus item shares the same
    label as the query.

    Args:
        query_embeddings: Array-like of shape (n_queries, embed_dim).
        corpus_embeddings: Array-like of shape (n_corpus, embed_dim).
        query_labels: Array-like of shape (n_queries,) with ground-truth labels.
        corpus_labels: Array-like of shape (n_corpus,) with ground-truth labels.
        k_values (Optional[List[int]]): List of K values for Recall@K.
            Defaults to [1, 5, 10].

    Returns:
        Dict[str, float]: Dictionary with 'mrr' and 'recall@K' for each K.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    query_embeddings = np.asarray(query_embeddings, dtype=np.float64)
    corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float64)
    query_labels = np.asarray(query_labels)
    corpus_labels = np.asarray(corpus_labels)

    # L2-normalize for cosine similarity via dot product
    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-8)
    query_normed = query_embeddings / q_norms

    c_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    c_norms = np.maximum(c_norms, 1e-8)
    corpus_normed = corpus_embeddings / c_norms

    # Similarity matrix: (n_queries, n_corpus)
    similarities = query_normed @ corpus_normed.T

    # Sort indices by descending similarity for each query
    sorted_indices = np.argsort(-similarities, axis=1)

    # Relevance matrix: does the corpus item share the query label?
    # Shape: (n_queries, n_corpus)
    relevance = (corpus_labels[sorted_indices] == query_labels[:, np.newaxis])

    # MRR: mean reciprocal rank of the first relevant item
    reciprocal_ranks = []
    for i in range(relevance.shape[0]):
        relevant_positions = np.where(relevance[i])[0]
        if len(relevant_positions) > 0:
            reciprocal_ranks.append(1.0 / (relevant_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    mrr = float(np.mean(reciprocal_ranks))

    # Recall@K
    results: Dict[str, float] = {"mrr": mrr}
    for k in k_values:
        top_k_relevant = relevance[:, :k].any(axis=1).astype(float)
        recall_at_k = float(np.mean(top_k_relevant))
        results[f"recall@{k}"] = recall_at_k

    logger.info(
        f"Retrieval evaluation: MRR={mrr:.4f}, "
        + ", ".join(f"Recall@{k}={results[f'recall@{k}']:.4f}" for k in k_values)
    )

    return results
