# main.py
import argparse
import concurrent
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset as data_load
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import LongformerTokenizer

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.evaluation.metrics import evaluate_text_prediction
from chiron.evaluation.visualization import plot_evaluation_metrics
from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel, create_adjacency_matrix
from chiron.pipeline import TextPredictionPipeline
from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train, evaluate
from chiron.utils.config import Config
from chiron.utils.data import SemanticFoldingDataset


def build_vocab(
    preprocessed_conversations: List[List[str]], max_vocab_size: int
) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    token_counts: Counter = count_token_frequencies(preprocessed_conversations)

    for idx, (token, count) in enumerate(
        token_counts.most_common(max_vocab_size - 2), start=2
    ):
        vocab[token] = idx

    return vocab


def main(config_path: str) -> None:
    """
    Main function to run the Semantic Folding training pipeline with k-fold cross-validation.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = Config(config_path)

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir="runs")

    # Preprocess text
    logger.info("Preprocessing text...")
    preprocessor = TextPreprocessor(**config["preprocessing_params"])

    conversations = load_dataset(config)
    # main.py
    preprocessed_conversations = preprocessor.preprocess(
        conversations, cache_key="preprocessed_conversations"
    )
    preprocessed_conversations_flattened = [
        token for conv in preprocessed_conversations for token in conv
    ]

    logger.info(
        f"Number of preprocessed conversations: {len(preprocessed_conversations)}"
    )
    logger.debug(f"First preprocessed conversation: {preprocessed_conversations[0]}")

    # Generate word embeddings
    logger.info("Generating word embeddings...")
    embedding_model = Word2VecEmbedding(**config["embedding_params"])
    embeddings = embedding_model.generate_embeddings(
        preprocessed_conversations_flattened, cache_key="embeddings"
    )
    logger.info(f"Number of embeddings: {len(embeddings)}")

    # Generate SDRs
    logger.info("Generating SDRs...")
    sdr_generator = SDRGenerator(**config["sdr_params"])
    sdr_embeddings = sdr_generator.generate_sdr_embeddings(embeddings)
    logger.info(f"SDR embeddings shape: {sdr_embeddings.shape}")

    # Create an adjacency matrix
    logger.info("Creating adjacency matrix...")
    device = torch.device(config["device"])
    sdr_embeddings_float = sdr_embeddings.astype(float)

    adjacency_matrix_sparse = create_adjacency_matrix(
        sdr_embeddings_float,
        threshold=config["adjacency_matrix"]["threshold"],
        subsample_rate=config["adjacency_matrix"]["subsample_rate"],
        n_jobs=config["adjacency_matrix"].get("n_jobs", -1),
        chunk_size=config["adjacency_matrix"].get("chunk_size", 5000),
        device=device,
    )

    # Load the tokenizer
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    # Create a vocabulary dictionary from the preprocessed conversations
    preprocessed_conversations = [
        [token] for conv in preprocessed_conversations for token in conv
    ]

    vocab = build_vocab(
        preprocessed_conversations, max_vocab_size=preprocessor.max_vocab_size
    )

    # Create dataset
    dataset = SemanticFoldingDataset(sdr_embeddings, tokenizer, labels=None)

    # Perform k-fold cross-validation
    k = config.get("k_folds", 5)
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    # Initialize fold_metrics with default values for each metric
    default_metrics = {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "loss": float("inf"),
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"Training fold {fold + 1}/{k}")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"] or 4,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"] or 4,
            pin_memory=True,
        )

        logger.info(f"Creating SNN model for fold {fold + 1}...")
        snn_model: Union[SNNModel, nn.DataParallel] = SNNModel(
            sp_params=config["sdr_params"],
            longformer_params=config["longformer_params"],
            gat_params=config["gat_params"],
            htm_params=config["htm_params"],
            device=device,
            vocab=vocab,
            tokenizer=tokenizer,
            max_sequence_length=config["max_sequence_length"],
            snn_params=config["snn_params"],
        ).to(device)

        # Wrap the model with DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            snn_model = nn.DataParallel(snn_model)

        # Train model
        train_config = {
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "accumulation_steps": config["accumulation_steps"],
            "patience": config.get("patience", 5),
        }

        logger.info(f"Training model for fold {fold + 1}...")
        train(
            snn_model,
            train_dataloader,
            val_dataloader,
            tokenizer,
            train_config,
            device,
            adjacency_matrix_sparse,
            writer,
        )

        # Evaluate model and plot metrics
        logger.info(f"Evaluating model for fold {fold + 1}...")
        val_metrics = evaluate(
            snn_model,
            val_dataloader,
            tokenizer,
            device,
            adjacency_matrix_sparse,
        )
        if not val_metrics:
            logger.warning(
                f"No validation metrics returned for fold {fold + 1}. Using default values."
            )
            val_metrics = default_metrics.copy()
        fold_metrics.append(val_metrics)

        # Evaluate text prediction performance
        logger.info(f"Evaluating text prediction for fold {fold + 1}...")
        text_prediction_metrics = evaluate_text_prediction(
            snn_model,
            tokenizer,
            val_dataset,
            device,
            max_length=config["text_prediction"]["max_length"],
            num_return_sequences=config["text_prediction"]["num_return_sequences"],
        )
        logger.info(
            f"Text prediction metrics for fold {fold + 1}: {text_prediction_metrics}"
        )

        # Compute average metrics across folds
        avg_metrics = {
            metric: np.mean(
                [fold.get(metric, default_metrics[metric]) for fold in fold_metrics]
            )
            for metric in default_metrics.keys()
        }

        logger.info("Average metrics across folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Create the text prediction pipeline
        pipeline = TextPredictionPipeline(
            snn_model,
            tokenizer,
            device,
            max_length=config["text_prediction"]["max_length"],
            num_return_sequences=config["text_prediction"]["num_return_sequences"],
        )

        # Perform text prediction
        input_text = "what is the meaning of life? let me know your thoughts"
        predicted_texts = pipeline(input_text)
        logger.info(f"Predicted texts: {predicted_texts}")

        if config["labels"]:
            semantic_sim_accuracy, semantic_sim_f1 = semantic_similarity_prediction(
                sdr_embeddings, config["labels"]
            )
            text_class_accuracy, text_class_f1 = text_classification(
                sdr_embeddings, config["labels"]
            )
        else:
            semantic_sim_accuracy, semantic_sim_f1 = 0.0, 0.0
            text_class_accuracy, text_class_f1 = 0.0, 0.0

        logger.info("Plotting evaluation metrics...")
        plot_evaluation_metrics(
            avg_metrics,
            semantic_sim_accuracy,
            semantic_sim_f1,
            text_class_accuracy,
            text_class_f1,
        )

        logger.info("Training completed.")

        # Close TensorBoard writer
        writer.close()


def load_dataset(config: Config) -> List[List[Dict[str, Any]]]:
    """
    Load the dataset based on the configuration.

    Args:
        config (Config): The configuration object.

    Returns:
        List[List[Dict[str, Any]]]: The loaded dataset as a list of conversations,
            where each conversation is a list of dictionaries representing individual turns.
    """
    dataset_params = config["dataset_params"]
    dataset_name = dataset_params["dataset_name"]
    dataset_config = dataset_params.get("dataset_config", None)
    split = dataset_params["split"]

    dataset = data_load(dataset_name, dataset_config, split=split)
    conversations = dataset["conversations"]

    return conversations


def count_token_frequencies(tokenized_texts: List[List[str]]) -> Counter:
    """
    Count token frequencies in the tokenized texts using ThreadPoolExecutor.

    Args:
        tokenized_texts (List[List[str]]): List of tokenized texts.

    Returns:
        Counter: Token frequency counts.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda: Counter(text))
            for text in tqdm(tokenized_texts, desc="submitting tasks...")
        ]

        token_counts = Counter()
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Counting token frequencies",
        ):
            token_counts.update(future.result())

    return token_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Folding Training")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )
    args = parser.parse_args()
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        level="INFO",
    )
    logger.add("logs/semantic_folding_{time}.log", serialize=True, level="DEBUG")
    main(config_path=args.config)
