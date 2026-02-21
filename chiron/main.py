import argparse
import os
import random
import sys
import time
from typing import Dict, List, Any, Tuple, Union

import neptune
import numpy as np
import scipy
import scipy.sparse
import torch
import torch.distributed as dist
from datasets import load_dataset as data_load
from loguru import logger
from scipy.sparse import csr_matrix, save_npz, load_npz
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
from chiron.train import train, collate_fn
from chiron.utils.config import Config
from chiron.utils.data import SemanticFoldingDataset

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# ---------------------------------------------------------------------------
# CUDA performance tuning for A100
# ---------------------------------------------------------------------------
# Enable TF32 for matmuls and convolutions on A100 (Ampere architecture).
# TF32 provides ~3x throughput vs fp32 with negligible precision loss for
# neural network training. Uses 19-bit mantissa (10 bits from fp16 + 9 bits
# from tf32 rounding) which preserves gradient entropy better than fp16.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark mode to auto-tune convolution algorithms
    # for the specific hardware and input shapes (caches optimal kernels).
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Optional FAISS import -- fall back to batched cosine if unavailable
# ---------------------------------------------------------------------------
try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss not found – falling back to batched pairwise cosine similarity. "
        "Install faiss-cpu or faiss-gpu for faster adjacency construction."
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_all_seeds(seed: int) -> None:
    """Set seeds for torch, numpy, and the stdlib random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"All random seeds set to {seed}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FAISS-based K-NN adjacency builder
# ---------------------------------------------------------------------------

def _build_adjacency_faiss(
    embeddings_np: np.ndarray,
    top_k: int,
    output_file: str,
) -> Tuple[csr_matrix, int]:
    """Build a sparse adjacency matrix using FAISS IndexFlatIP (inner product on
    L2-normalised vectors ≡ cosine similarity).

    Returns:
        (csr_matrix, num_embeddings)
    """
    n, d = embeddings_np.shape
    k = min(top_k, n)  # cannot have more neighbours than points

    logger.info(
        f"FAISS adjacency: n={n}, d={d}, top_k={k}"
    )

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(d)
    index.add(embeddings_np)

    logger.info("FAISS index built – querying K nearest neighbours …")
    # D = similarities, I = indices  (both shape [n, k])
    D, I = index.search(embeddings_np, k)

    # Stream directly into COO arrays – never materialise a dense (n, n) matrix
    row_indices = np.repeat(np.arange(n, dtype=np.int64), k)
    col_indices = I.ravel().astype(np.int64)
    data = np.ones(len(row_indices), dtype=np.int8)

    # Remove self-loops (optional – keeps diagonal clean)
    mask = row_indices != col_indices
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    data = data[mask]

    adjacency = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n, n)
    )

    save_npz(output_file, adjacency)
    logger.info(
        f"FAISS adjacency matrix saved to {output_file}  "
        f"(nnz={adjacency.nnz}, density={adjacency.nnz / (n * n):.6f})"
    )
    return adjacency, n


# ---------------------------------------------------------------------------
# Batched cosine-similarity fallback (original approach, kept for compat)
# ---------------------------------------------------------------------------

def _build_adjacency_batched(
    sdr_embeddings: torch.Tensor,
    threshold: float,
    batch_size: int,
    output_file: str,
    device: torch.device,
    chunk_size: int,
) -> Tuple[csr_matrix, int]:
    """Original O(n²) batched pairwise cosine + threshold approach."""
    sdr_embeddings = sdr_embeddings.to(device)
    sdr_embeddings_normalized = torch.nn.functional.normalize(sdr_embeddings, p=2, dim=1)
    num_embeddings = sdr_embeddings.shape[0]

    row_indices: List[np.ndarray] = []
    col_indices: List[np.ndarray] = []
    data: List[np.ndarray] = []

    for start_idx in tqdm(
        range(0, num_embeddings, chunk_size), desc="Computing adjacency matrix (fallback)"
    ):
        end_idx = min(start_idx + chunk_size, num_embeddings)
        chunk_embeddings = sdr_embeddings_normalized[start_idx:end_idx]

        for batch_start_idx in range(0, chunk_embeddings.shape[0], batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, chunk_embeddings.shape[0])
            batch_embeddings = chunk_embeddings[batch_start_idx:batch_end_idx]

            similarity_matrix = torch.einsum("ij,kj->ik", batch_embeddings, chunk_embeddings)
            batch_adjacency_matrix = similarity_matrix >= threshold

            batch_row_indices, batch_col_indices = torch.where(batch_adjacency_matrix)
            batch_data = torch.ones(batch_row_indices.shape[0], dtype=torch.int8)

            batch_row_indices += batch_start_idx
            batch_col_indices += start_idx

            row_indices.append(batch_row_indices.cpu().numpy())
            col_indices.append(batch_col_indices.cpu().numpy())
            data.append(batch_data.cpu().numpy())

    row_all = np.concatenate(row_indices)
    col_all = np.concatenate(col_indices)
    data_all = np.concatenate(data)

    adjacency = csr_matrix(
        (data_all, (row_all, col_all)), shape=(num_embeddings, num_embeddings)
    )
    save_npz(output_file, adjacency)
    logger.info(f"Adjacency matrix (fallback) saved to: {output_file}")
    return adjacency, num_embeddings


# ---------------------------------------------------------------------------
# Public adjacency API (preserves original signature)
# ---------------------------------------------------------------------------

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
    top_k: int = 50,
) -> Tuple[Union[scipy.sparse.csr_matrix, torch.Tensor], int]:
    """
    Compute and save the adjacency matrix.

    Primary path: FAISS-based approximate K-NN graph (O(n·k) memory, fast).
    Fallback: batched pairwise cosine similarity with fixed threshold when
    FAISS is not installed.

    Args:
        sdr_embeddings (torch.Tensor): The SDR embeddings tensor.
        threshold (float): Similarity threshold (used only in fallback mode).
        batch_size (int): Batch size for fallback computation.
        output_file (str): Path to save the adjacency matrix (.npz).
        device (torch.device): Device for fallback computation.
        fallback_mode (str): Legacy fallback strategy ("subsample" | "cluster").
        subsample_ratio (float): Legacy – subsample ratio for fallback.
        num_clusters (int): Legacy – cluster count for fallback.
        eps (float): Legacy – DBSCAN eps for fallback.
        min_samples (int): Legacy – DBSCAN min_samples for fallback.
        chunk_size (int): Chunk size for fallback batched computation.
        top_k (int): Number of nearest neighbours for K-NN graph (FAISS path).

    Returns:
        Tuple[Union[csr_matrix, torch.Tensor], int]:
            The computed sparse adjacency matrix and the number of embeddings.
    """
    t0 = time.perf_counter()

    if _FAISS_AVAILABLE:
        logger.info(f"Using FAISS K-NN adjacency builder (top_k={top_k})")
        embeddings_np = sdr_embeddings.detach().cpu().numpy().astype(np.float32)
        adjacency, n = _build_adjacency_faiss(embeddings_np, top_k, output_file)
    else:
        logger.info("Using batched cosine-similarity fallback for adjacency matrix")
        adjacency, n = _build_adjacency_batched(
            sdr_embeddings, threshold, batch_size, output_file, device, chunk_size
        )

    elapsed = time.perf_counter() - t0
    logger.info(f"Adjacency matrix construction completed in {elapsed:.1f}s")
    return adjacency, n


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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """
    Main function to run the Semantic Folding training pipeline with k-fold cross-validation.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = Config(config_path)

    # ---- Reproducibility ----
    seed = config.get("seed", 42)
    _set_all_seeds(seed)

    # ---- Logging / tracking ----
    logger.info("=" * 60)
    logger.info("Chiron – Semantic Folding Training Pipeline")
    logger.info("=" * 60)

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
        "seed": seed,
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "weight_decay": config.get("weight_decay", 0.01),
        "label_smoothing": config.get("label_smoothing", 0.1),
        "ema_decay": config.get("ema_decay", 0.999),
    }

    # ---- Stage 1: Text Preprocessing ----
    logger.info("[Stage 1/6] Preprocessing text …")
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

    # ---- Stage 2: Tokenizer ----
    logger.info("[Stage 2/6] Loading tokenizer …")
    tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["name"])

    # ---- Stage 3: Word Embeddings ----
    logger.info("[Stage 3/6] Generating word embeddings …")
    embedding_model = Word2VecEmbedding(**config["embedding_params"])
    embeddings = embedding_model.generate_embeddings(
        preprocessed_conversations, cache_key="embeddings"
    )
    logger.info(f"Number of embeddings: {len(embeddings)}")

    # ---- Stage 4: SDR Generation ----
    logger.info("[Stage 4/6] Generating SDRs …")
    sdr_generator = SDRGenerator(**config["sdr_params"])
    sdr_embeddings: torch.Tensor = torch.tensor(
        sdr_generator.generate_sdr_embeddings(embeddings), dtype=torch.float32
    )
    logger.info(f"SDR embeddings shape: {sdr_embeddings.shape}")

    # Update the sp_params dictionary with the sdr_embeddings tensor
    config["sdr_params"]["sdr_embeddings"] = sdr_embeddings

    # ---- Stage 5: Adjacency Matrix ----
    logger.info("[Stage 5/6] Computing adjacency matrix …")
    device = torch.device(config["device"])

    adjacency_matrix_file = config["adjacency_matrix"]["output_file"]
    adj_cfg = config["adjacency_matrix"]
    compute_and_save_adjacency_matrix(
        sdr_embeddings,
        threshold=adj_cfg.get("threshold", 0.5),
        batch_size=adj_cfg.get("batch_size", 256),
        output_file=adjacency_matrix_file,
        device=device,
        top_k=adj_cfg.get("top_k", 50),
    )
    adjacency_matrix = load_adjacency_matrix(adjacency_matrix_file)

    # ---- Stage 6: K-Fold Training ----
    logger.info("[Stage 6/6] Starting K-fold cross-validation …")
    dataset = SemanticFoldingDataset(sdr_embeddings, tokenizer)

    k = config.get("k_folds", 5)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    # ---- DataLoader throughput configuration ----
    # num_workers: 4 per GPU is optimal for A100 (avoids CPU contention)
    # prefetch_factor: pre-load 4 batches per worker into pinned memory
    # persistent_workers: keep worker processes alive between epochs (avoids
    #   fork/spawn overhead, saves ~2-3s per epoch)
    # pin_memory: allocate batch tensors in CUDA-pinned (page-locked) memory
    #   for faster async CPU→GPU transfers via non_blocking=True
    _num_workers = config["num_workers"] or 4
    _prefetch_factor = 4  # Load 4 batches ahead per worker

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"Training fold {fold + 1}/{k}")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=_prefetch_factor,
            collate_fn=collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=_prefetch_factor,
            collate_fn=collate_fn,
        )

        logger.info(f"Creating SNN model for fold {fold + 1}...")

        snn_model = SNNModel(
            sp_params=config["sdr_params"],
            gat_params=config["gat_params"],
            htm_params=config["htm_params"],
            snn_params=config["snn_params"],
            device=device,
        ).to(device)

        # ---- Multi-GPU strategy ----
        # DataParallel has ~2x memory overhead (replicates model per GPU) and
        # uses a single-process GIL-bottlenecked approach.
        # For 4xA100, torch.compile provides better single-GPU utilization
        # through kernel fusion without the replication overhead.
        # For true multi-GPU, users should launch with torchrun for DDP.
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            snn_model = nn.DataParallel(snn_model)

        # ---- torch.compile for kernel fusion ----
        # Fuses small operations (LayerNorm, dropout, activations) into single
        # CUDA kernels, reducing kernel launch overhead by ~40% and improving
        # memory bandwidth utilization on A100. Uses inductor backend which
        # generates Triton kernels optimized for the specific GPU architecture.
        try:
            snn_model = torch.compile(snn_model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without compilation: {e}")

        # Train model
        train_config = {
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "accumulation_steps": config["accumulation_steps"],
            "patience": config.get("patience", 5),
            "warmup_ratio": config.get("warmup_ratio", 0.1),
            "weight_decay": config.get("weight_decay", 0.01),
            "label_smoothing": config.get("label_smoothing", 0.1),
            "ema_decay": config.get("ema_decay", 0.999),
        }

        logger.info(f"Training model for fold {fold + 1}...")

        train_losses, val_losses = train(
            snn_model,
            train_dataloader,
            val_dataloader,
            tokenizer,
            train_config,
            device,
            adjacency_matrix,  # Pre-cached on GPU inside train()
            writer,
            checkpoint_dir=".checkpoints",
        )

        # Log training and validation losses to Neptune
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            run[f"fold_{fold + 1}/train/loss"].append(train_loss)
            run[f"fold_{fold + 1}/val/loss"].append(val_loss)

    logger.info("Pipeline complete.")


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
