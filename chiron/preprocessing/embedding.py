# embedding.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from chiron.utils.cache import load_cached_data, cache_data


class Word2VecEmbedding:
    """
    Class for generating Word2Vec embeddings.
    """

    def __init__(
        self,
        vector_size: int,
        window: int,
        min_count: int,
        workers: int,
        model_path: str,
    ):
        """
        Initialize the Word2VecEmbedding instance.

        Args:
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the model.
            model_path (str): Path to the pre-trained Word2Vec model file.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        # Load the pre-trained Word2Vec model
        try:
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Word2Vec model file not found at: {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading Word2Vec model: {str(e)}")

        # Check if the loaded model has the expected vector size
        assert (
            self.model.vector_size == vector_size
        ), f"Loaded model has vector size {self.model.vector_size}, but expected {vector_size}"

    def generate_embeddings(
        self,
        preprocessed_conversations: List[str],
        cache_key: str,
        batch_size: int = 1000,
    ) -> List[np.ndarray]:
        """
        Generate Word2Vec embeddings for the given preprocessed conversations using threading and batch processing.

        Args:
            preprocessed_conversations (List[str]): List of preprocessed conversations.
            cache_key (str): Key to use for caching the generated embeddings.
            batch_size (int): Number of conversations to process in each batch (default: 1000).

        Returns:
            List[np.ndarray]: List of word embeddings for each conversation.
        """
        embeddings = load_cached_data(cache_key)
        if embeddings is None:
            embeddings: list = self._generate_embeddings_parallel(
                preprocessed_conversations, batch_size
            )
            cache_data(embeddings, cache_key)
        return embeddings

    def _generate_embeddings_parallel(
        self, tokenized_texts: List[str], batch_size: int
    ) -> List[np.ndarray]:
        """
        Generate Word2Vec embeddings in parallel using multithreading.

        Args:
            tokenized_texts (List[str]): List of tokenized texts.
            batch_size (int): Number of texts to process in each batch.

        Returns:
            List[np.ndarray]: List of word embeddings for each text.
        """
        total_texts = len(tokenized_texts)
        batches = [
            tokenized_texts[i : i + batch_size]
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

    def _process_batch(self, batch: List[str]) -> List[np.ndarray]:
        """
        Process a batch of tokenized texts and generate embeddings.

        Args:
            batch (List[str]): Batch of tokenized texts.

        Returns:
            List[np.ndarray]: List of word embeddings for each text in the batch.
        """
        batch_embeddings = []
        for text in batch:
            # Check if token is a string and exists in the Word2Vec model
            if isinstance(text, str) and text in self.model.key_to_index:
                batch_embeddings.append(self.model.get_vector(text))
            else:
                # Handle unknown tokens or non-string tokens
                # Use a default embedding of zeros
                default_embedding = np.zeros(self.vector_size, dtype=np.float32)
                batch_embeddings.append(default_embedding)

        return batch_embeddings
