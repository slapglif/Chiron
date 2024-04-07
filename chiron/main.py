import argparse
import os
import sys
from typing import Dict, List, Any

import neptune
from datasets import load_dataset as data_load
from scipy.sparse import load_npz
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel
from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train, collate_fn
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


import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
from loguru import logger
from sklearn.cluster import KMeans, DBSCAN
import numpy as np


def compute_and_save_adjacency_matrix(
        sdr_embeddings: torch.Tensor,
        threshold: float = 0.5,
        batch_size: int = 256,
        output_file: str = "adjacency_matrix.npz",
        device: torch.device = torch.device("cuda:0"),
        fallback_mode: str = "subsample",
        subsample_ratio: float = 0.01,
        num_clusters: int = 100,
        eps: float = 0.5,
        min_samples: int = 10,
        chunk_size: int = 10000,
) -> None:
    # Move the SDR embeddings tensor to the specified device
    sdr_embeddings = sdr_embeddings.to(device)

    # Normalize the SDR embeddings
    sdr_embeddings_normalized = torch.nn.functional.normalize(
        sdr_embeddings, p=2, dim=1
    )

    num_embeddings = sdr_embeddings.shape[0]

    if fallback_mode == "subsample":
        # Perform subsampling
        subsample_size = int(num_embeddings * subsample_ratio)
        subsample_indices = np.random.choice(num_embeddings, size=subsample_size, replace=False)
        sdr_embeddings_normalized = sdr_embeddings_normalized[subsample_indices]
        num_embeddings = subsample_size
    elif fallback_mode == "cluster":
        # Perform clustering
        if num_embeddings <= 10000:
            clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
        else:
            clustering_model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_labels = clustering_model.fit_predict(sdr_embeddings_normalized.cpu().numpy())
        unique_labels = np.unique(cluster_labels)
        num_clusters = len(unique_labels)

    # Create a sparse matrix in COO format
    row_indices = []
    col_indices = []
    data = []

    if fallback_mode == "cluster":
        for cluster_label in unique_labels:
            cluster_mask = cluster_labels == cluster_label
            cluster_embeddings = sdr_embeddings_normalized[cluster_mask]

            for start_idx in range(0, cluster_embeddings.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, cluster_embeddings.shape[0])
                batch_embeddings = cluster_embeddings[start_idx:end_idx]

                # Compute the similarity within the current batch using einsum
                similarity_matrix = torch.einsum('ij,kj->ik', batch_embeddings, cluster_embeddings)

                # Apply the threshold to create the adjacency matrix for the current batch
                batch_adjacency_matrix = similarity_matrix >= threshold

                # Convert the batch adjacency matrix to COO format
                batch_row_indices, batch_col_indices = torch.where(batch_adjacency_matrix)
                batch_data = torch.ones(batch_row_indices.shape[0], dtype=torch.int8)

                # Adjust the row and column indices based on the cluster and batch
                batch_row_indices += start_idx
                batch_col_indices = torch.tensor([np.where(cluster_mask)[0][i] for i in batch_col_indices])

                # Append the batch indices and data to the overall adjacency matrix
                row_indices.append(batch_row_indices.cpu().numpy())
                col_indices.append(batch_col_indices.cpu().numpy())
                data.append(batch_data.cpu().numpy())
    else:
        for start_idx in tqdm(range(0, num_embeddings, chunk_size), desc="Computing adjacency matrix"):
            end_idx = min(start_idx + chunk_size, num_embeddings)
            chunk_embeddings = sdr_embeddings_normalized[start_idx:end_idx]

            for batch_start_idx in range(0, chunk_embeddings.shape[0], batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, chunk_embeddings.shape[0])
                batch_embeddings = chunk_embeddings[batch_start_idx:batch_end_idx]

                # Compute the similarity within the current batch using einsum
                similarity_matrix = torch.einsum('ij,kj->ik', batch_embeddings, chunk_embeddings)

                # Apply the threshold to create the adjacency matrix for the current batch
                batch_adjacency_matrix = similarity_matrix >= threshold

                # Convert the batch adjacency matrix to COO format
                batch_row_indices, batch_col_indices = torch.where(batch_adjacency_matrix)
                batch_data = torch.ones(batch_row_indices.shape[0], dtype=torch.int8)

                # Adjust the row indices based on the current batch and chunk
                batch_row_indices += batch_start_idx
                batch_col_indices += start_idx

                # Append the batch indices and data to the overall adjacency matrix
                row_indices.append(batch_row_indices.cpu().numpy())
                col_indices.append(batch_col_indices.cpu().numpy())
                data.append(batch_data.cpu().numpy())

    # Concatenate the row indices, column indices, and data
    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)
    data = np.concatenate(data)

    # Create a sparse matrix in CSR format
    adjacency_matrix_sparse = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(num_embeddings, num_embeddings)
    )

    # Save the adjacency matrix to a file
    save_npz(output_file, adjacency_matrix_sparse)
    logger.info(f"Adjacency matrix saved to: {output_file}")

    # Return the adjacency matrix and the original number of embeddings
    return adjacency_matrix_sparse, num_embeddings


def load_adjacency_matrix(file_path: str) -> csr_matrix:
    """
    Load the adjacency matrix from a file.

    Args:
        file_path (str): The path to the file containing the adjacency matrix.

    Returns:
        csr_matrix: The loaded adjacency matrix.
    """
    adjacency_matrix = load_npz(file_path)
    return adjacency_matrix


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
    sdr_embeddings: torch.Tensor = torch.tensor(
        sdr_generator.generate_sdr_embeddings(embeddings), dtype=torch.float32
    )
    logger.info(f"SDR embeddings shape: {sdr_embeddings.shape}")

    # Create an adjacency matrix
    logger.info("Creating adjacency matrix...")
    device = torch.device(config["device"])

    # Compute and save the adjacency matrix
    logger.info("Computing and saving adjacency matrix...")
    adjacency_matrix_file = config["adjacency_matrix"]["output_file"]
    compute_and_save_adjacency_matrix(
        sdr_embeddings,
        threshold=config["adjacency_matrix"]["threshold"],
        batch_size=config["adjacency_matrix"]["batch_size"],
        output_file=adjacency_matrix_file,
        device=device,
    )
    adjacency_matrix = load_adjacency_matrix(adjacency_matrix_file)

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["name"])

    # Create dataset
    dataset = SemanticFoldingDataset(sdr_embeddings, tokenizer)

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
            persistent_workers=True,
            collate_fn=collate_fn,  # Use the collate_fn from the train module
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"] or 4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,  # Use the collate_fn from the train module
        )

        logger.info(f"Creating SNN model for fold {fold + 1}...")
        snn_model = SNNModel(
            sp_params=config["sdr_params"],
            gat_params=config["gat_params"],
            htm_params=config["htm_params"],
            snn_params=config["snn_params"],
            device=device,
        ).to(device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            snn_model = nn.DataParallel(snn_model)

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
            adjacency_matrix,  # Pass the SciPy sparse matrix
            writer,
            checkpoint_dir=".checkpoints",
        )

        # Log training and validation losses to Neptune
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            run[f"fold_{fold + 1}/train/loss"].append(train_loss)
            run[f"fold_{fold + 1}/val/loss"].append(val_loss)


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

    # Create the checkpoint directory if it doesn't exist
    os.makedirs(".checkpoints", exist_ok=True)

    main(config_path=args.config)
