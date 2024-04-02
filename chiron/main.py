# main.py

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Any

import neptune
import numpy as np
import torch
from datasets import load_dataset as data_load
from loguru import logger
from scipy.sparse import coo_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel
from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train
from chiron.utils.config import Config
from chiron.utils.data import SemanticFoldingDataset

os.environ["JOBLIB_MULTIPROCESSING"] = "0"


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


def build_vocab(
        preprocessed_conversations: List[str], max_vocab_size: int
) -> Dict[str, int]:
    """
    Build a vocabulary from the preprocessed conversations.

    Args:
        preprocessed_conversations (List[str]): List of preprocessed conversations.
        max_vocab_size (int): Maximum size of the vocabulary.

    Returns:
        Dict[str, int]: Vocabulary dictionary mapping tokens to indices.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    token_counts: Counter = TextPreprocessor.count_token_frequencies(
        preprocessed_conversations
    )

    for idx, (token, count) in enumerate(
            token_counts.most_common(max_vocab_size - 2), start=2
    ):
        vocab[token] = idx

    return vocab


def lsh_cosine_similarity(
        sdr_embeddings: torch.Tensor,
        threshold: float = 0.5,
        num_projections: int = 128,
        seed: int = 42,
        chunk_size: int = 1000,  # Number of embeddings to process at a time
        batch_size: int = 10000,  # Number of rows to process in each batch
) -> torch.Tensor:
    num_embeddings = sdr_embeddings.size(0)
    device = sdr_embeddings.device

    # Normalize embeddings to unit vectors for cosine similarity calculation
    sdr_embeddings_normalized = sdr_embeddings / sdr_embeddings.norm(
        dim=1, keepdim=True
    )

    # Convert embeddings to numpy array
    sdr_embeddings_numpy = sdr_embeddings_normalized.cpu().numpy()

    # Create an imputer transformer
    imputer = SimpleImputer(strategy="mean")

    # Impute missing values
    sdr_embeddings_imputed = imputer.fit_transform(sdr_embeddings_numpy)

    # Create random projection matrix for LSH
    random_projection = SparseRandomProjection(
        n_components=num_projections, random_state=seed
    )

    # Project embeddings using random projection
    projected_embeddings = random_projection.fit_transform(sdr_embeddings_imputed)

    # Compute hash codes for projected embeddings
    hash_codes = np.sign(projected_embeddings).astype(np.int32)
    hash_codes = np.ascontiguousarray(hash_codes)  # Ensure C-contiguity

    # Initialize lists to store the sparse matrix indices and values
    row_indices = []
    col_indices = []
    values = []

    # Process the adjacency matrix in batches
    for i in tqdm(range(0, num_embeddings, batch_size), desc="Processing batches"):
        start = i
        end = min(i + batch_size, num_embeddings)

        # Compute Hamming distances and populate the adjacency matrix in chunks
        for j in range(start, end, chunk_size):
            chunk_start = j
            chunk_end = min(j + chunk_size, end)

            # Process the embeddings in chunks
            for k in range(chunk_start, chunk_end):
                # Compute Hamming distances using memory-efficient operations
                hamming_distances_chunk = np.sum(
                    hash_codes[k] != hash_codes[start:end], axis=1
                )
                estimated_cosine_similarity_chunk = 1 - hamming_distances_chunk / (
                        num_projections * 2
                )

                # Threshold the estimated cosine similarity and store the indices and values
                mask = estimated_cosine_similarity_chunk >= threshold
                row_indices.extend([k] * np.count_nonzero(mask))
                col_indices.extend(np.where(mask)[0] + start)
                values.extend(estimated_cosine_similarity_chunk[mask])

        # Create a sparse matrix for the current batch
        batch_adjacency_matrix = coo_matrix(
            (values, (row_indices, col_indices)), shape=(num_embeddings, num_embeddings)
        )

        # Convert the batch adjacency matrix to a sparse COO tensor
        indices = (
            torch.from_numpy(
                np.vstack((batch_adjacency_matrix.row, batch_adjacency_matrix.col))
            )
            .long()
            .to(device)
        )
        values = torch.from_numpy(batch_adjacency_matrix.data).float().to(device)
        batch_adjacency_matrix_tensor = torch.sparse_coo_tensor(
            indices, values, (num_embeddings, num_embeddings), device=device
        )

        # Yield the batch adjacency matrix tensor
        yield batch_adjacency_matrix_tensor

        # Clear the lists for the next batch
        row_indices.clear()
        col_indices.clear()
        values.clear()


def create_adjacency_matrix_lsh(
        sdr_embeddings: torch.Tensor,
        threshold: float = 0.5,
        num_projections: int = 128,
        seed: int = 42,
) -> torch.sparse.Tensor:
    adjacency_matrix_sparse = lsh_cosine_similarity(
        sdr_embeddings, threshold, num_projections, seed
    )
    return adjacency_matrix_sparse


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

    # Initialize Neptune run
    run = neptune.init_run(
        project="AnunaChiron/Chiron",
        api_token=config.get("neptune_key"),
    )

    # Log hyperparameters
    run["parameters"] = {
        "preprocessing_params": config["preprocessing_params"],
        "embedding_params": config["embedding_params"],
        "sdr_params": config["sdr_params"],
        "adjacency_matrix": config["adjacency_matrix"],
        "gat_params": config["gat_params"],
        "htm_params": config["htm_params"],
        "snn_params": config["snn_params"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "num_epochs": config["num_epochs"],
        "learning_rate": config["learning_rate"],
        "accumulation_steps": config["accumulation_steps"],
        "patience": config.get("patience", 5),
    }

    # Preprocess text
    logger.info("Preprocessing text...")
    preprocessor = TextPreprocessor(**config["preprocessing_params"])

    conversations = load_dataset(config)
    preprocessed_conversations = preprocessor.preprocess(
        conversations, cache_key="preprocessed_conversations"
    )

    logger.info(
        f"Number of preprocessed conversations: {len(preprocessed_conversations)}"
    )
    logger.debug(f"First preprocessed conversation: {preprocessed_conversations[0]}")

    # Flatten the preprocessed conversations
    preprocessed_conversations = [
        token for conv in preprocessed_conversations for token in conv
    ]

    logger.info(f"Number of preprocessed tokens: {len(preprocessed_conversations)}")
    logger.debug(f"First preprocessed token: {preprocessed_conversations[0]}")

    # Generate word embeddings
    logger.info("Generating word embeddings...")
    embedding_model = Word2VecEmbedding(**config["embedding_params"])
    embeddings = embedding_model.generate_embeddings(
        preprocessed_conversations, cache_key="embeddings"
    )
    logger.info(f"Number of embeddings: {len(embeddings)}")

    # Generate SDRs
    logger.info("Generating SDRs...")
    sdr_generator = SDRGenerator(**config["sdr_params"])
    sdr_embeddings: np.ndarray = sdr_generator.generate_sdr_embeddings(embeddings)
    logger.info(f"SDR embeddings shape: {sdr_embeddings.shape}")

    # Create an adjacency matrix
    logger.info("Creating adjacency matrix...")
    device = torch.device(config["device"])
    sdr_embeddings_tensor = torch.from_numpy(sdr_embeddings).float().to(device)

    # Create an adjacency matrix using LSH
    logger.info("Creating adjacency matrix using LSH...")
    adjacency_matrix_sparse = create_adjacency_matrix_lsh(
        sdr_embeddings_tensor,
        threshold=config["adjacency_matrix"]["threshold"],
        num_projections=config["adjacency_matrix"]["num_projections"],
        seed=config["adjacency_matrix"]["seed"],
    )

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["name"])

    # Create a vocabulary dictionary from the preprocessed conversations
    vocab = build_vocab(
        preprocessed_conversations, max_vocab_size=preprocessor.max_vocab_size
    )

    # Create dataset
    dataset = SemanticFoldingDataset(sdr_embeddings, tokenizer, labels=None)

    # Perform k-fold cross-validation
    k = config.get("k_folds", 5)
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

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
        snn_model = SNNModel(
            sp_params=config["sdr_params"],
            gat_params=config["gat_params"],
            htm_params=config["htm_params"],
            device=device,
            vocab=vocab,
            tokenizer=tokenizer,
            snn_params=config["snn_params"],
        ).to(device)

        # Train model
        train_config = {
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "accumulation_steps": config["accumulation_steps"],
            "patience": config.get("patience", 5),
        }

        logger.info(f"Training model for fold {fold + 1}...")
        train_losses, val_losses = train(
            snn_model,
            train_dataloader,
            val_dataloader,
            tokenizer,
            train_config,
            device,
            adjacency_matrix_sparse,
            writer,
        )

        # Log training and validation losses to Neptune
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            run[f"fold_{fold + 1}/train/loss"].append(train_loss)
            run[f"fold_{fold + 1}/val/loss"].append(val_loss)

    # Close TensorBoard writer
    writer.close()

    # Stop the Neptune run
    run.stop()


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
