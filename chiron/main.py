import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Any, Generator
from typing import Optional

import neptune
import torch
from datasets import load_dataset as data_load
from loguru import logger
from sklearn.model_selection import KFold
from torch import nn
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


def compute_top_k_cosine_similarity(
    batch_embeddings: torch.Tensor,
    sdr_embeddings_normalized: torch.Tensor,
    top_k: int,
    device: torch.device,
    similarity_batch_size: int = 512,
) -> torch.Tensor:
    """
    Compute the top-k cosine similarity for each embedding in the batch.

    Args:
        batch_embeddings (torch.Tensor): The batch of embeddings.
        sdr_embeddings_normalized (torch.Tensor): The normalized SDR embeddings.
        top_k (int): The number of top similar embeddings to consider.
        device (torch.device): The device to use for tensor operations.
        similarity_batch_size (int): The batch size for computing cosine similarity.

    Returns:
        torch.Tensor: The top-k cosine similarity values and indices for each embedding in the batch.
    """
    num_sdr_embeddings = sdr_embeddings_normalized.size(0)
    num_similarity_batches = (
        num_sdr_embeddings + similarity_batch_size - 1
    ) // similarity_batch_size

    top_k_values = torch.zeros((batch_embeddings.size(0), top_k), device=device)
    top_k_indices = torch.zeros(
        (batch_embeddings.size(0), top_k), dtype=torch.long, device=device
    )

    for i in range(num_similarity_batches):
        start_idx = i * similarity_batch_size
        end_idx = min((i + 1) * similarity_batch_size, num_sdr_embeddings)
        similarity_batch = sdr_embeddings_normalized[start_idx:end_idx]

        # Compute the cosine similarity between the batch embeddings and the current similarity batch
        similarity_matrix = torch.nn.functional.cosine_similarity(
            batch_embeddings.unsqueeze(1), similarity_batch.unsqueeze(0), dim=2
        )

        # Update the top-k values and indices
        batch_top_k_values, batch_top_k_indices = torch.topk(
            similarity_matrix, k=top_k, dim=1
        )
        mask = batch_top_k_values > top_k_values
        top_k_values[mask] = batch_top_k_values[mask]
        top_k_indices[mask] = batch_top_k_indices[mask] + start_idx

    return top_k_values, top_k_indices


def adjacency_matrix_generator(
    sdr_embeddings: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 100,
    top_k: int = 1000,
    device: torch.device = torch.device("cpu"),
    similarity_batch_size: int = 1000,
) -> torch.Tensor:
    """
    Generate batches of the adjacency matrix.

    Args:
        sdr_embeddings (torch.Tensor): The SDR embeddings tensor.
        threshold (float): The cosine similarity threshold for considering two embeddings as neighbors.
        batch_size (int): The batch size for computing the adjacency matrix.
        top_k (int): The number of top similar embeddings to consider for each embedding.
        device (torch.device): The device to use for tensor operations (e.g., torch.device("cuda") for GPU).
        similarity_batch_size (int): The batch size for computing cosine similarity.

    Yields:
        torch.Tensor: A batch of the adjacency matrix as a sparse tensor.
    """
    num_embeddings = sdr_embeddings.shape[0]
    num_batches = (num_embeddings + batch_size - 1) // batch_size

    # Move the SDR embeddings tensor to the specified device
    sdr_embeddings = sdr_embeddings.to(device)

    # Normalize the SDR embeddings
    sdr_embeddings_normalized = torch.nn.functional.normalize(
        sdr_embeddings, p=2, dim=1
    )

    for i in tqdm(range(num_batches), desc="Generating adjacency matrix batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_embeddings)
        batch_embeddings = sdr_embeddings_normalized[start_idx:end_idx]

        # Compute the top-k cosine similarity for the current batch
        top_k_values, top_k_indices = compute_top_k_cosine_similarity(
            batch_embeddings,
            sdr_embeddings_normalized,
            top_k,
            device,
            similarity_batch_size,
        )

        # Apply the threshold to create the adjacency matrix for the current batch
        adjacency_matrix_batch = (top_k_values >= threshold).float()

        # Create the sparse adjacency matrix batch
        indices = torch.stack(
            (
                torch.arange(start_idx, end_idx, device=device)
                .unsqueeze(1)
                .expand_as(top_k_indices),
                top_k_indices,
            ),
            dim=0,
        ).view(2, -1)
        values = adjacency_matrix_batch.view(-1)
        adjacency_matrix_batch = torch.sparse_coo_tensor(
            indices, values, (num_embeddings, num_embeddings), device=device
        )

        yield adjacency_matrix_batch


def compute_and_save_adjacency_matrix(
    sdr_embeddings: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 100,
    top_k: int = 1000,
    output_file_prefix: str = "adjacency_matrix_batch_",
    device: torch.device = torch.device("cpu"),
    similarity_batch_size: int = 1000,
) -> None:
    """
    Compute the adjacency matrix from SDR embeddings using cosine similarity and save it in batches.

    Args:
        sdr_embeddings (torch.Tensor): The SDR embeddings tensor.
        threshold (float): The cosine similarity threshold for considering two embeddings as neighbors.
        batch_size (int): The batch size for computing the adjacency matrix.
        top_k (int): The number of top similar embeddings to consider for each embedding.
        output_file_prefix (str): The prefix for the output file names of the adjacency matrix batches.
        device (torch.device): The device to use for tensor operations (e.g., torch.device("cuda") for GPU).
        similarity_batch_size (int): The batch size for computing cosine similarity.
    """
    # Compute the adjacency matrix in batches
    adjacency_matrix_batches = adjacency_matrix_generator(
        sdr_embeddings, threshold, batch_size, top_k, device, similarity_batch_size
    )

    # Save each batch of the adjacency matrix to a separate file
    for i, batch in enumerate(adjacency_matrix_batches):
        output_file = f"{output_file_prefix}{i}.pt"
        torch.save(batch, output_file)
        logger.info(f"Adjacency matrix batch {i} saved to: {output_file}")


def load_adjacency_matrix_batches(
    output_file_prefix: str, num_batches: int, device: Optional[torch.device] = None
) -> Generator[torch.Tensor, None, None]:
    """
    Load the adjacency matrix batches from files.

    Args:
        output_file_prefix (str): The prefix for the output file names of the adjacency matrix batches.
        num_batches (int): The number of batches to load.
        device (torch.device, optional): The device to load the batches on. If not provided, the batches
            will be loaded on the same device as they were saved.

    Yields:
        torch.Tensor: The loaded adjacency matrix batches.
    """
    for i in range(num_batches):
        output_file = f"{output_file_prefix}{i}.pt"
        batch = torch.load(output_file, map_location=device)
        yield batch


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
    sdr_embeddings: torch.Tensor = torch.tensor(
        sdr_generator.generate_sdr_embeddings(embeddings), dtype=torch.float32
    )
    logger.info(f"SDR embeddings shape: {sdr_embeddings.shape}")

    # Create an adjacency matrix
    logger.info("Creating adjacency matrix...")
    device = torch.device(config["device"])

    # Compute and save the adjacency matrix
    logger.info("Computing and saving adjacency matrix...")
    compute_and_save_adjacency_matrix(
        sdr_embeddings,
        threshold=config["adjacency_matrix"]["threshold"],
        batch_size=config["adjacency_matrix"]["batch_size"],
        device=device,
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
            persistent_workers=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"] or 4,
            pin_memory=True,
            persistent_workers=True,
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
        scaler = (
            torch.cuda.amp.GradScaler()
        )  # Initialize the GradScaler for mixed precision training

        train_losses, val_losses = train(
            snn_model,
            train_dataloader,
            val_dataloader,
            tokenizer,
            train_config,
            device,  # Pass the output directory for loading adjacency matrix chunks
            writer,
            scaler,  # Pass the GradScaler to the train function
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
