from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from chiron.utils.cache import load_cached_data, cache_data


class Word2VecEmbedding:
    """Class for generating Word2Vec embeddings."""

    def __init__(self, vector_size: int, window: int, min_count: int, workers: int):
        """
        Initialize the Word2VecEmbedding instance.

        Args:
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the model.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = KeyedVectors.load_word2vec_format(
            ".w2v_models/GoogleNews-vectors-negative300.bin", binary=True
        )

    def generate_embeddings(
            self, tokenized_texts: List[List[str]], cache_key: str, batch_size: int = 1000
    ) -> List[List[np.ndarray]]:
        """
        Generate Word2Vec embeddings for the given tokenized texts using threading and batch processing.

        Args:
            tokenized_texts (List[List[str]]): List of tokenized texts.
            cache_key (str): Key to use for caching the generated embeddings.
            batch_size (int): Number of texts to process in each batch (default: 1000).

        Returns:
            List[List[np.ndarray]]: List of word embeddings for each text.
        """
        embeddings = load_cached_data(cache_key)
        if embeddings is None:
            embeddings: list = self._generate_embeddings_parallel(
                tokenized_texts, batch_size
            )
            cache_data(embeddings, cache_key)
        return embeddings

    def _generate_embeddings_parallel(
            self, tokenized_texts: List[List[str]], batch_size: int
    ) -> List[List[np.ndarray]]:
        """
        Generate Word2Vec embeddings in parallel using multithreading.

        Args:
            tokenized_texts (List[List[str]]): List of tokenized texts.
            batch_size (int): Number of texts to process in each batch.

        Returns:
            List[List[np.ndarray]]: List of word embeddings for each text.
        """
        total_texts = len(tokenized_texts)
        batches = [
            tokenized_texts[i: i + batch_size]
            for i in range(0, total_texts, batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self._process_batch, batch)
                for batch in tqdm(batches, desc="submitting tasks...")
            ]
            embeddings = [
                future.result()
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Generating word embedding batches...",
                )
            ]

        embeddings = [embedding for batch in embeddings for embedding in batch]
        return embeddings

    def _process_batch(self, batch: List[List[str]]) -> List[List[np.ndarray]]:
        """
        Process a batch of tokenized texts and generate embeddings.

        Args:
            batch (List[List[str]]): Batch of tokenized texts.

        Returns:
            List[List[np.ndarray]]: List of word embeddings for each text in the batch.
        """
        batch_embeddings = []
        for text in batch:
            text_embeddings = [
                self.model[token]
                for token in text
                if token in self.model
            ]
            batch_embeddings.append(text_embeddings)
        return batch_embeddings
