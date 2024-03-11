import numpy as np
from loguru import logger
from sklearn.decomposition import IncrementalPCA
from typing import List
from torch.utils.tensorboard import SummaryWriter
import os

from tqdm import tqdm


class SDRGenerator:
    """Class for generating Sparse Distributed Representations (SDRs) from word embeddings."""

    def __init__(
        self,
        pca_components: int,
        sdr_dimensions: int,
        sparsity: float,
        batch_size: int = 1000,
        log_dir: str = "logs",
    ):
        """
        Initialize the SDRGenerator instance.

        Args:
            pca_components (int): Number of components for PCA dimensionality reduction.
            sdr_dimensions (int): Number of dimensions for the SDR embeddings.
            sparsity (float): Sparsity level for the SDR embeddings.
            batch_size (int): Batch size for incremental PCA.
            log_dir (str): Directory to store TensorBoard logs.
        """
        self.pca_components = pca_components
        self.sdr_dimensions = sdr_dimensions
        self.sparsity = sparsity
        self.batch_size = batch_size
        self.pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)
        self.projection_matrix = None
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info("Initialized SDRGenerator with the following parameters:")
        logger.info(f"  PCA Components: {pca_components}")
        logger.info(f"  SDR Dimensions: {sdr_dimensions}")
        logger.info(f"  Sparsity: {sparsity}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Log Directory: {log_dir}")

    def generate_sdr_embeddings(self, embeddings: List[List[np.ndarray]]) -> np.ndarray:
        """
        Generate Sparse Distributed Representations (SDRs) from word embeddings.

        Args:
            embeddings (List[List[np.ndarray]]): List of word embeddings.

        Returns:
            np.ndarray: SDR embeddings.
        """
        logger.info("Generating SDR embeddings...")
        valid_embeddings = []
        for batch_embeddings in tqdm(embeddings, desc="Embedding batches"):
            batch_embeddings = [embedding for embedding in batch_embeddings if len(embedding) > 0]
            if len(batch_embeddings) > 0:
                valid_embeddings.extend(batch_embeddings)

        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings found.")

        total_valid_embeddings = len(valid_embeddings)
        logger.info(f"Total number of valid embeddings: {total_valid_embeddings}")
        self.writer.add_scalar("Embeddings/Valid", total_valid_embeddings)

        # Adjust n_components based on the number of valid embeddings
        n_components = min(self.pca_components, total_valid_embeddings)
        self.pca.n_components = n_components

        embeddings = self.apply_pca(valid_embeddings)
        sdr_embeddings = self.binarize(embeddings)
        logger.info(f"Generated {len(sdr_embeddings)} SDR embeddings.")
        self.writer.add_scalar("Embeddings/SDR", len(sdr_embeddings))
        return sdr_embeddings

    def apply_pca(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Apply PCA dimensionality reduction to the word embeddings.

        Args:
            embeddings (List[np.ndarray]): Word embeddings.

        Returns:
            np.ndarray: PCA-reduced embeddings.
        """
        logger.info("Applying PCA dimensionality reduction...")
        for i in tqdm(range(0, len(embeddings), self.batch_size), desc="Applying PCA dimensionality reduction"):
            batch = embeddings[i:i + self.batch_size]
            self.pca.partial_fit(batch)
        return self.pca.transform(embeddings)

    def binarize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binarize the embeddings to create SDRs.

        Args:
            embeddings (np.ndarray): PCA-reduced embeddings.

        Returns:
            np.ndarray: SDR embeddings.
        """
        logger.info("Binarizing embeddings to create SDRs...")
        if self.projection_matrix is None:
            self.projection_matrix = np.random.randn(
                embeddings.shape[1], self.sdr_dimensions
            )
        projected_embeddings = np.dot(embeddings, self.projection_matrix)
        sorted_indices = np.argsort(projected_embeddings, axis=1)
        num_active_bits = int(self.sparsity * self.sdr_dimensions)
        sdr_embeddings = np.zeros((len(embeddings), self.sdr_dimensions), dtype=bool)
        for i in tqdm(range(len(embeddings)), desc="Binarizing embeddings"):
            active_indices = sorted_indices[i, -num_active_bits:]
            sdr_embeddings[i, active_indices] = True
        logger.info("Binarization completed.")
        self.writer.add_scalar("Embeddings/SDR_Sparsity", np.mean(sdr_embeddings))
        self.writer.add_histogram("Embeddings/SDR_Distribution", sdr_embeddings)
        return sdr_embeddings

    def save_model(self, save_path: str):
        """
        Save the SDRGenerator model.

        Args:
            save_path (str): Path to save the model.
        """
        logger.info(f"Saving SDRGenerator model to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            pca_components=self.pca.components_,
            projection_matrix=self.projection_matrix,
        )
        logger.info("Model saved successfully.")

    def load_model(self, load_path: str):
        """
        Load the SDRGenerator model.

        Args:
            load_path (str): Path to load the model from.
        """
        logger.info(f"Loading SDRGenerator model from {load_path}")
        loaded_data = np.load(load_path)
        self.pca.components_ = loaded_data["pca_components"]
        self.projection_matrix = loaded_data["projection_matrix"]
        logger.info("Model loaded successfully.")
