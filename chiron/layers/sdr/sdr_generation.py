import numpy as np
from sklearn.decomposition import TruncatedSVD
from loguru import logger
from typing import List, Optional, Union

from tqdm import tqdm


class SDRGenerator:
    """
    Generates Sparse Distributed Representations (SDRs) from high-dimensional data vectors.
    Utilizes dimensionality reduction and binary thresholding to produce SDRs.

    Attributes:
        projection_dimensions (int): Target dimensionality after dimensionality reduction.
        sdr_dimensions (int): Final dimensionality of the SDRs.
        sparsity (float): Fraction of bits that are active in the SDR.
    """

    def __init__(
        self,
        projection_dimensions: int,
        sdr_dimensions: int,
        sparsity: float,
        use_gpu: bool = True,
    ) -> None:
        """
        Initializes the SDRGenerator with the given parameters.

        Args:
            projection_dimensions (int): The number of dimensions to reduce the input embeddings to, should be less than sdr_dimensions.
            sdr_dimensions (int): The number of dimensions in the resulting SDR.
            sparsity (float): The proportion of active bits in the SDR, typically a small value (e.g., 0.05 for 5% sparsity).
        """
        self.projection_dimensions = projection_dimensions
        self.sdr_dimensions = sdr_dimensions
        self.sparsity = sparsity

        # Ensure that the projection dimension is less than the SDR dimensions
        if self.projection_dimensions >= self.sdr_dimensions:
            logger.error("Projection dimensions should be less than SDR dimensions.")
            raise ValueError(
                "Projection dimensions should be less than SDR dimensions."
            )

        self.projection = TruncatedSVD(n_components=self.projection_dimensions)

    def generate_sdr_embeddings(
        self, embeddings: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Generates SDR embeddings from the provided high-dimensional data vectors.

        Args:
            embeddings (List[np.ndarray]): A list of high-dimensional data vectors to be converted to SDRs.

        Returns:
            Optional[np.ndarray]: An array of SDRs or None if the process fails.
        """
        if not embeddings:
            logger.warning("No embeddings provided for SDR generation.")
            return None

        # Filter out empty or invalid embeddings and validate data
        valid_embeddings = [
            emb
            for emb in tqdm(embeddings, desc="Filtering Invalid Embeddings...")
            if emb.size > 0 and not np.isnan(emb).any() and not np.isinf(emb).any()
        ]

        if len(valid_embeddings) == 0:
            logger.warning("No valid embeddings found after filtering.")
            return None

        # Concatenate the list of arrays into a single 2D array for SVD
        concatenated_embeddings = np.vstack(valid_embeddings)

        try:
            logger.info("Reducing Embedding Dimensionality...")
            # Dimensionality reduction using SVD
            reduced_embeddings = self.projection.fit_transform(concatenated_embeddings)

            # Binarize the reduced embeddings to generate SDRs
            logger.info("Binarize the reduced embeddings...")
            threshold = np.percentile(reduced_embeddings, (1 - self.sparsity) * 100)
            sdr_embeddings = []
            for embedding in tqdm(reduced_embeddings, desc="Generating SDRs"):
                sdr_embedding = (embedding >= threshold).astype(int)
                sdr_embeddings.append(sdr_embedding)

            return np.array(sdr_embeddings)

        except Exception as e:
            logger.error(f"An unexpected error occurred during SDR generation: {e}")

        return None
