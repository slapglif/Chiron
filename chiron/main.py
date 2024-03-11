import argparse
import sys
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.evaluation.visualization import plot_evaluation_metrics
from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel, create_adjacency_matrix

from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train, evaluate
from chiron.utils.cache import load_cached_data, cache_data
from chiron.utils.config import Config


def convert_sdr_embeddings_to_list(sdr_embeddings: np.ndarray) -> List[List[float]]:
    """
    Convert SDR embeddings from numpy array to list of lists.

    Args:
        sdr_embeddings (np.ndarray): SDR embeddings as a numpy array.

    Returns:
        List[List[float]]: SDR embeddings as a list of lists.
    """
    sdr_list: list | object = sdr_embeddings.tolist()
    return sdr_list


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
        # cache_data(data_dict, dataset_key + f"_{i}")
        logger.info(f"Loaded dataset with {len(data)} conversations")

        # Preprocess text
        preprocessed_data_key = f"preprocessed_data_{dataset_key}"
        preprocessed_data = load_cached_data(preprocessed_data_key)

        logger.info("Preprocessing text...")
        if not preprocessed_data:
            preprocessor = TextPreprocessor(**config["preprocessing_params"])
            preprocessed_data = preprocessor.preprocess(
                data, cache_key=preprocessed_data_key
            )
            logger.info(f"Preprocessed {len(preprocessed_data)} conversations")
        else:
            logger.info(
                f"Loaded cached preprocessed data with {len(preprocessed_data)} conversations"
            )

        # Generate word embeddings
        embeddings_key = f"embeddings_{preprocessed_data_key}_{config['embedding_params']['vector_size']}_{config['embedding_params']['window']}_{config['embedding_params']['min_count']}_{config['embedding_params']['workers']}"  # noqa: E501
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
            logger.info(
                f"Loaded cached word embeddings with {len(embeddings)} embeddings"
            )

        # Generate SDRs
        sdr_embeddings_key = f"sdr_embeddings_{embeddings_key}_{config['sdr_params']['pca_components']}_{config['sdr_params']['sdr_dimensions']}_{config['sdr_params']['sparsity']}"  # noqa: E501
        sdr_embeddings_list = []
        logger.info("Generating SDRs...")
        for _i in range(0, len(embeddings), config["batch_size"]):
            batch_embeddings = embeddings[_i : _i + config["batch_size"]]
            batch_sdr_embeddings_key = f"{sdr_embeddings_key}_{_i}"
            batch_sdr_embeddings = load_cached_data(batch_sdr_embeddings_key)
            if not batch_sdr_embeddings:
                sdr_generator = SDRGenerator(**config["sdr_params"])
                batch_sdr_embeddings = sdr_generator.generate_sdr_embeddings(
                    batch_embeddings
                )
                cache_data(batch_sdr_embeddings, batch_sdr_embeddings_key)
                logger.info(
                    f"Generated {len(batch_sdr_embeddings)} SDR embeddings for batch {_i}"  # noqa: E501
                )
            else:
                logger.info(f"Loaded cached SDR embeddings for batch {_i}")
            sdr_embeddings_list.extend(batch_sdr_embeddings)

            # Create SNN model
            logger.info("Creating SNN model...")
            snn_model = SNNModel(
                gat_params=config["gat_params"],
                htm_params=config["htm_params"],
                encoder_params=config["encoder_params"],
                decoder_params=config["decoder_params"],
                snn_params=config["snn_params"],
                sp_params=config["htm_params"],
            )
            logger.info(f"Created SNN model: {snn_model}")

            # Train model
            logger.info("Training model...")
            device = config["device"]
            train_config = {
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
                "learning_rate": config["learning_rate"],
            }
            train(
                snn_model,
                sdr_embeddings_list,
                train_config,
                device,
            )
            logger.info("Model training completed")

            # Create data loader for evaluation
            logger.info("Creating data loader for evaluation...")
            sdr_embeddings_tensor = torch.FloatTensor(sdr_embeddings_list).to(device)
            adjacency_matrix_tensor = torch.FloatTensor(create_adjacency_matrix(sdr_embeddings_list)).to(device)  # noqa: E501
            dataset = TensorDataset(sdr_embeddings_tensor, adjacency_matrix_tensor)
            dataloader = DataLoader(dataset, batch_size=config["batch_size"])
            logger.info(f"Created data loader with {len(dataloader)} batches")

            # Evaluate model and plot metrics
            logger.info("Evaluating model...")
            evaluation_scores: Dict[str, float] = evaluate(
                snn_model, dataloader, device
            )  # noqa: E501
            logger.info(f"Evaluation scores: {evaluation_scores}")

            semantic_sim_accuracy, semantic_sim_f1 = semantic_similarity_prediction(
                sdr_embeddings_list, config["labels"]
            )
            logger.info(
                f"Semantic similarity prediction - Accuracy: {semantic_sim_accuracy:.4f}, F1: {semantic_sim_f1:.4f}"  # noqa: E501
            )

            text_class_accuracy, text_class_f1 = text_classification(
                sdr_embeddings_list, config["labels"]
            )
            logger.info(
                f"Text classification - Accuracy: {text_class_accuracy:.4f}, F1: {text_class_f1:.4f}"  # noqa: E501
            )

            logger.info("Plotting evaluation metrics...")
            plot_evaluation_metrics(
                evaluation_scores,
                semantic_sim_accuracy,
                semantic_sim_f1,
                text_class_accuracy,
                text_class_f1,
            )
            logger.info("Evaluation metrics plotted")

            logger.success("Semantic Folding training pipeline completed successfully!")


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
