import asyncio
import sys
from typing import Dict, List, Any

import argparse
import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.evaluation.visualization import plot_evaluation_metrics
from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel, create_adjacency_matrix_parallel
from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train, evaluate
from chiron.utils.cache import load_cached_data, cache_data
from chiron.utils.config import Config


def convert_sdr_embeddings_to_list(sdr_embeddings: np.ndarray) -> object:
    """
    Convert SDR embeddings from numpy array to list of lists.

    Args:
        sdr_embeddings (np.ndarray): SDR embeddings as a numpy array.

    Returns:
        List[List[float]]: SDR embeddings as a list of lists.
    """
    sdr_list = sdr_embeddings.tolist()
    return sdr_list


async def build_sdr_embeddings_list(
        i: int,
        embeddings: List[Any],
        config: Dict[str, Any],
        sdr_embeddings_key: str,
        sdr_embeddings_list: List[Any],
) -> None:
    """
    Build SDR embeddings list in an asynchronous manner.

    Args:
        i (int): Index for slicing embeddings.
        embeddings (List[Any]): List of embeddings.
        config (Dict[str, Any]): Configuration dictionary.
        sdr_embeddings_key (str): Key for caching SDR embeddings.
        sdr_embeddings_list (List[Any]): List to extend with SDR embeddings.
    """
    batch_embeddings = embeddings[i: i + config["batch_size"]]
    batch_sdr_embeddings_key = f"{sdr_embeddings_key}_{i}"
    batch_sdr_embeddings = load_cached_data(batch_sdr_embeddings_key)
    if not batch_sdr_embeddings:
        sdr_generator = SDRGenerator(**config["sdr_params"])
        batch_sdr_embeddings = sdr_generator.generate_sdr_embeddings(batch_embeddings)
        cache_data(batch_sdr_embeddings, batch_sdr_embeddings_key)
    else:
        logger.info(f"Loaded cached SDR embeddings for batch {i}")
    sdr_embeddings_list.extend(batch_sdr_embeddings)


def generate_labels(sdr_embeddings: np.ndarray) -> np.ndarray:
    """
    Generate labels dynamically based on SDR embeddings.

    Args:
        sdr_embeddings (np.ndarray): SDR embeddings array.

    Returns:
        np.ndarray: Generated labels array.
    """
    # Example: Generate labels based on the sum of active bits in each SDR embedding
    active_bits_sum = np.sum(sdr_embeddings, axis=1)

    # Define threshold values for label assignment
    threshold_low = np.percentile(active_bits_sum, 33.33)
    threshold_high = np.percentile(active_bits_sum, 66.67)

    # Assign labels based on the thresholds
    labels = np.where(active_bits_sum < threshold_low, "low", "medium")
    labels = np.where(active_bits_sum >= threshold_high, "high", labels)

    return labels


@logger.catch
def main(config_path: str) -> None:
    """
    Main function to run the Semantic Folding training pipeline.

    Args:
        config_path (str): Path to the configuration file.
    """
    logger.info("Starting Semantic Folding training pipeline...")

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = Config(config_path)
    device = config["device"]

    # Load dataset
    dataset_key = f"dataset_{config['dataset_params']['dataset_name']}_{config['dataset_params']['dataset_config']}_{config['dataset_params']['split']}"
    data = load_cached_data(dataset_key)
    if not data:
        logger.info("Loading dataset...")
        dataset = load_dataset(
            config["dataset_params"]["dataset_name"],
            config["dataset_params"]["dataset_config"],
            split=config["dataset_params"]["split"],
            streaming=False,
            cache_dir=".cache",
        )

        data = dataset["conversations"]
        logger.info(f"Loaded dataset with {len(data)} conversations")

        # Preprocess text
        preprocessed_data_key = f"preprocessed_data_{dataset_key}"
        preprocessed_data = load_cached_data(preprocessed_data_key)

        logger.info("Preprocessing text...")
        if not preprocessed_data:
            preprocessor = TextPreprocessor(**config["preprocessing_params"])
            preprocessed_data = preprocessor.preprocess(data, cache_key=preprocessed_data_key)
            logger.info(f"Preprocessed {len(preprocessed_data)} conversations")
        else:
            logger.info(f"Loaded cached preprocessed data with {len(preprocessed_data)} conversations")

        # Generate word embeddings
        embeddings_key = f"embeddings_{preprocessed_data_key}_{config['embedding_params']}"  # noqa: E501
        embeddings = load_cached_data(embeddings_key)
        logger.info("Generating word embeddings...")
        if not embeddings:
            embedding_model = Word2VecEmbedding(**config["embedding_params"])
            embeddings = embedding_model.generate_embeddings(
                preprocessed_data,
                cache_key=embeddings_key,
                batch_size=config["batch_size"],
            )
            logger.info(f"Generated {len(embeddings)} word embeddings")
        else:
            logger.info(f"Loaded cached word embeddings with {len(embeddings)} embeddings")

        # Generate SDR embeddings
        sdr_embeddings_key = f"sdr_embeddings_{embeddings_key}_{config['sdr_params']}"
        sdr_embeddings = load_cached_data(sdr_embeddings_key)
        logger.info("Generating SDR embeddings...")
        if not sdr_embeddings:
            sdr_embeddings_list = []
            for i in tqdm(range(0, len(embeddings), config["batch_size"])):
                asyncio.run(
                    build_sdr_embeddings_list(
                        i,
                        embeddings,
                        config.config,
                        sdr_embeddings_key,
                        sdr_embeddings_list,
                    )
                )

            sdr_embeddings = np.array(sdr_embeddings_list)
            logger.info(f"Generated {len(sdr_embeddings)} SDR embeddings")
        else:
            logger.info(f"Loaded cached SDR embeddings with {len(sdr_embeddings)} embeddings")

        # Generate labels dynamically
        logger.info("Generating labels dynamically...")
        labels = generate_labels(sdr_embeddings)

        # Convert embeddings to a NumPy array
        embeddings = np.array(embeddings)

        # Ensure all tensors have the same size along the first dimension
        num_samples = min(sdr_embeddings.shape[0], embeddings.shape[0], labels.shape[0])
        sdr_embeddings = sdr_embeddings[:num_samples]
        embeddings = embeddings[:num_samples]
        labels = labels[:num_samples]

        # Create adjacency matrix
        adjacency_matrix_key = f"adjacency_matrix_{sdr_embeddings_key}_{config['snn_params']}"
        adjacency_matrix = load_cached_data(adjacency_matrix_key)
        logger.info("Creating adjacency matrix...")
        if not adjacency_matrix:
            adjacency_matrix = create_adjacency_matrix_parallel(
                sdr_embeddings,
                n_jobs=config["adjacency_matrix"]["n_jobs"],
            )
            logger.info(f"Created adjacency matrix of shape {adjacency_matrix.shape}")
        else:
            logger.info(f"Loaded cached adjacency matrix of shape {adjacency_matrix.shape}")

        # Create SNN model
        logger.info("Creating SNN model...")
        snn_model = SNNModel(
            sp_params={},
            snn_params=config["snn_params"],
            gat_params=config["gat_params"],
            encoder_params=config["encoder_params"],
            decoder_params=config["decoder_params"],
            htm_params=config["htm_params"],
        ).to(device)
        logger.info(f"Created SNN model {snn_model}")

        # Prepare data for training
        logger.info("Preparing data for training...")
        sdr_embeddings_tensor = torch.tensor(sdr_embeddings, dtype=torch.float32)

        # Convert adjacency matrix to sparse COO tensor
        nonzero_indices = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)
        values = torch.tensor(adjacency_matrix.data, dtype=torch.float32)
        adjacency_matrix_tensor = torch.sparse_coo_tensor(
            nonzero_indices,
            values,
            adjacency_matrix.shape
        )

       # embedded_conversations_tensor = torch.tensor(embeddings, dtype=torch.float32)

        # Create labels tensor
        labels_tensor = torch.tensor(np.vectorize(lambda x: {"low": 0, "medium": 1, "high": 2}[x])(labels),
                                     dtype=torch.long)

        # Train SNN model
        logger.info("Training SNN model...")
        train(
            model=snn_model,
            sdr_embeddings=sdr_embeddings_tensor,
            adjacency_matrix=adjacency_matrix_tensor,
            conversation_texts=preprocessed_data,
            labels=labels_tensor,
            config=config["train_params"],
            device=device,
        )

        # Evaluate SNN model
        logger.info("Evaluating SNN model...")
        sdr_list = convert_sdr_embeddings_to_list(sdr_embeddings)
        evaluation_metrics = evaluate(snn_model,
                                      DataLoader(TensorDataset(sdr_embeddings_tensor), batch_size=config["batch_size"]),
                                      adjacency_matrix_tensor, device)
        logger.info(f"Evaluation metrics: {evaluation_metrics}")

        # Perform downstream tasks
        logger.info("Performing downstream tasks...")
        semantic_similarity_results = semantic_similarity_prediction(sdr_list, labels)
        logger.info(f"Semantic similarity results: {semantic_similarity_results}")

        text_classification_results = text_classification(sdr_list, labels)
        logger.info(f"Text classification results: {text_classification_results}")

        # Visualize evaluation metrics
        logger.info("Visualizing evaluation metrics...")
        plot_evaluation_metrics(
            evaluation_metrics,
            semantic_similarity_results,
            text_classification_results,
            config["visualization_params"],
            text_class_f1=text_classification_results["f1_score"],
        )

        logger.info("Semantic Folding training pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Folding Training")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )
    args = parser.parse_args()

    logger.remove()  # Remove the default logger
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
    )
    logger.add("logs/semantic_folding_{time}.log", serialize=True)

    main(args.config)