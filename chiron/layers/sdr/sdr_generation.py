# sdr_generation.py
import concurrent.futures
from typing import List

import numpy as np
from loguru import logger
from sklearn.decomposition import IncrementalPCA
from torch.utils.tensorboard import SummaryWriter


class SDRGenerator:
    """
    Class for generating Sparse Distributed Representations (SDRs) from embeddings.
    """

    def __init__(
        self,
        pca_components: int,
        sdr_dimensions: int,
        sparsity: float,
        batch_size: int = 1000,
        log_dir: str = "logs",
        num_workers: int = 4,
    ):
        """
        Initialize the SDRGenerator.

        Args:
            pca_components (int): Number of PCA components to retain.
            sdr_dimensions (int): Dimensions of the SDR vectors.
            sparsity (float): Sparsity level of the SDR vectors.
            batch_size (int): Batch size for processing embeddings.
            log_dir (str): Directory to store log files.
            num_workers (int): Number of worker processes for parallel processing.
        """
        self.pca_components = pca_components
        self.sdr_dimensions = sdr_dimensions
        self.sparsity = sparsity
        self.batch_size = batch_size
        self.pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)
        self.projection_matrix = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_workers = num_workers

    def generate_sdr_embeddings(self, embeddings: List[List[np.ndarray]]) -> np.ndarray:
        """
        Generate SDR embeddings from the given embeddings.

        Args:
            embeddings (List[List[np.ndarray]]): List of embeddings.

        Returns:
            np.ndarray: Generated SDR embeddings.
        """
        valid_embeddings = [
            emb
            for batch_embeddings in embeddings
            for emb in batch_embeddings
            if len(emb) > 0
        ]

        if not valid_embeddings:
            raise ValueError("No valid embeddings found.")

        total_valid_embeddings = len(valid_embeddings)

        n_components = min(self.pca_components, total_valid_embeddings)
        self.pca.n_components = n_components

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self.pca.partial_fit, batch)
                for batch in np.array_split(
                    valid_embeddings, max(1, len(valid_embeddings) // self.batch_size)
                )
            ]
            concurrent.futures.wait(futures)
            pca_embeddings = list(
                executor.map(
                    self.pca.transform,
                    np.array_split(
                        valid_embeddings,
                        max(1, len(valid_embeddings) // self.batch_size),
                    ),
                )
            )
            pca_embeddings = np.concatenate(pca_embeddings)

        sdr_embeddings = self.binarize(pca_embeddings)

        logger.debug("Initialized SDRGenerator with the following parameters:")
        logger.debug(f"  PCA Components: {self.pca_components}")
        logger.debug(f"  SDR Dimensions: {self.sdr_dimensions}")
        logger.debug(f"  Sparsity: {self.sparsity}")
        logger.debug(f"  Batch Size: {self.batch_size}")
        logger.debug(f"  Log Directory: {self.writer.get_logdir()}")
        logger.debug(f"  Number of Workers: {self.num_workers}")
        logger.debug(f"Total number of valid embeddings: {total_valid_embeddings}")
        logger.debug(f"Generated {len(sdr_embeddings)} SDR embeddings.")

        sparsity = np.mean(sdr_embeddings)
        logger.debug(f"SDR Sparsity: {sparsity:.4f}")

        self.writer.add_scalar("Embeddings/Valid", total_valid_embeddings)
        self.writer.add_scalar("Embeddings/SDR", len(sdr_embeddings))
        self.writer.add_scalar("Embeddings/SDR_Sparsity", sparsity)
        self.writer.add_histogram("Embeddings/SDR_Distribution", sdr_embeddings)

        return sdr_embeddings

    def binarize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binarize the embeddings to create SDRs.

        Args:
            embeddings (np.ndarray): Input embeddings.

        Returns:
            np.ndarray: Binarized SDR embeddings.
        """
        if self.projection_matrix is None:
            self.projection_matrix = np.random.randn(
                embeddings.shape[1], self.sdr_dimensions
            )
        projected_embeddings = np.dot(embeddings, self.projection_matrix)
        sorted_indices = np.argsort(projected_embeddings, axis=1)
        num_active_bits = int(self.sparsity * self.sdr_dimensions)
        sdr_embeddings = np.zeros((len(embeddings), self.sdr_dimensions), dtype=bool)

        for i in range(len(embeddings)):
            active_indices = sorted_indices[i, -num_active_bits:]
            sdr_embeddings[i, active_indices] = True

        return sdr_embeddings
