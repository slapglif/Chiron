# embedding.py

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from gensim.models import KeyedVectors
from loguru import logger
from tqdm import tqdm

from chiron.utils.cache import load_cached_data, cache_data


def _process_batch_worker(args: tuple) -> List[np.ndarray]:
    """
    Top-level worker function for ProcessPoolExecutor (must be picklable).

    Processes a batch of tokenized texts and generates embeddings with OOV fallbacks.

    Args:
        args: Tuple of (batch, model_path, vector_size, normalize, vocab_sample_vectors)
            - batch: List of tokens to embed
            - model_path: Path to the Word2Vec model file (reloaded per-process)
            - vector_size: Expected vector dimensionality
            - normalize: Whether to L2-normalize output embeddings
            - mean_vector: The precomputed mean embedding vector for final OOV fallback
    """
    batch, model_path, vector_size, normalize, mean_vector = args

    # Each worker process loads its own copy of the model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    mean_vector = np.array(mean_vector, dtype=np.float32)

    batch_embeddings = []
    for text in batch:
        if isinstance(text, str):
            embedding = _get_embedding_with_fallbacks(
                text, model, vector_size, mean_vector
            )
        else:
            embedding = mean_vector.copy()

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        batch_embeddings.append(embedding)

    return batch_embeddings


def _get_embedding_with_fallbacks(
    token: str,
    model: KeyedVectors,
    vector_size: int,
    mean_vector: np.ndarray,
) -> np.ndarray:
    """
    Get an embedding for a token with multi-level OOV fallback strategy:

    1. Direct lookup in the Word2Vec model vocabulary
    2. Subword decomposition: split the token into subwords and average their vectors
    3. Character n-gram hashing (FastText-style): hash character n-grams to model indices
    4. Final fallback: use the precomputed mean embedding (better than zeros)

    Args:
        token: The word token to embed.
        model: The loaded Word2Vec KeyedVectors model.
        vector_size: The expected vector dimensionality.
        mean_vector: The precomputed mean embedding for final fallback.

    Returns:
        np.ndarray: The embedding vector of shape (vector_size,).
    """
    # Level 1: Direct vocabulary lookup
    if token in model.key_to_index:
        return model.get_vector(token)

    # Level 2: Subword fallback - try splitting into meaningful subwords
    subword_embedding = _subword_fallback(token, model, vector_size)
    if subword_embedding is not None:
        return subword_embedding

    # Level 3: Character n-gram hashing fallback (FastText-style)
    ngram_embedding = _char_ngram_fallback(token, model, vector_size)
    if ngram_embedding is not None:
        return ngram_embedding

    # Level 4: Return mean embedding (better than zero vector)
    return mean_vector.copy()


def _subword_fallback(
    token: str, model: KeyedVectors, vector_size: int
) -> Optional[np.ndarray]:
    """
    Attempt to construct an embedding by decomposing the token into subwords.

    Strategy:
      - Try common prefixes/suffixes (e.g., "un-", "-ing", "-tion", "-ly", etc.)
      - Try splitting camelCase or hyphenated words
      - Average the vectors of any recognized subword parts

    Args:
        token: The OOV token.
        model: The Word2Vec model.
        vector_size: Expected dimensionality.

    Returns:
        The averaged subword embedding, or None if no subwords are found.
    """
    subwords = []

    # Split on hyphens, underscores, and camelCase boundaries
    import re

    parts = re.split(r"[-_]", token)
    # Further split camelCase
    expanded_parts = []
    for part in parts:
        camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
        expanded_parts.extend(camel_split)

    # Also try common English affixes
    affixes = [
        ("un", ""), ("re", ""), ("pre", ""), ("dis", ""), ("mis", ""),
        ("", "ing"), ("", "tion"), ("", "sion"), ("", "ment"),
        ("", "ness"), ("", "able"), ("", "ible"), ("", "ful"),
        ("", "less"), ("", "ly"), ("", "er"), ("", "est"),
        ("", "ed"), ("", "es"), ("", "s"),
    ]

    # Collect recognized subwords from split parts
    vectors = []
    for part in expanded_parts:
        part_lower = part.lower()
        if part_lower and part_lower in model.key_to_index:
            vectors.append(model.get_vector(part_lower))

    # If no split parts matched, try stripping affixes from the original token
    if not vectors:
        token_lower = token.lower()
        for prefix, suffix in affixes:
            stem = token_lower
            if prefix and stem.startswith(prefix):
                stem = stem[len(prefix):]
            if suffix and stem.endswith(suffix):
                stem = stem[: -len(suffix)]
            if stem and len(stem) >= 2 and stem in model.key_to_index:
                vectors.append(model.get_vector(stem))
                break  # Use the first successful affix stripping

    if vectors:
        return np.mean(vectors, axis=0).astype(np.float32)

    return None


def _char_ngram_fallback(
    token: str, model: KeyedVectors, vector_size: int, min_n: int = 3, max_n: int = 6
) -> Optional[np.ndarray]:
    """
    Generate an embedding using character n-gram hashing, similar to FastText.

    Extracts character n-grams from the token, hashes each to an index in the
    model's vocabulary, and averages the corresponding vectors.

    Args:
        token: The OOV token.
        model: The Word2Vec model.
        vector_size: Expected dimensionality.
        min_n: Minimum n-gram length (default: 3).
        max_n: Maximum n-gram length (default: 6).

    Returns:
        The averaged n-gram embedding, or None if no valid n-grams produce vectors.
    """
    # Add boundary markers like FastText
    padded = f"<{token.lower()}>"
    vocab_keys = model.index_to_key
    vocab_size = len(vocab_keys)

    if vocab_size == 0:
        return None

    vectors = []
    seen_indices = set()

    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            ngram = padded[i : i + n]
            # Hash the n-gram to a vocabulary index
            hash_val = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
            idx = hash_val % vocab_size

            if idx not in seen_indices:
                seen_indices.add(idx)
                vectors.append(model.get_vector(vocab_keys[idx]))

    if vectors:
        return np.mean(vectors, axis=0).astype(np.float32)

    return None


class Word2VecEmbedding:
    """
    Class for generating Word2Vec embeddings with enhanced OOV handling.

    Improvements over the original implementation:
        - Subword decomposition fallback for OOV tokens
        - Character n-gram hashing fallback (FastText-style)
        - Mean embedding fallback instead of zero vectors
        - Optional L2 normalization of output embeddings
        - ProcessPoolExecutor for CPU-bound embedding lookups (avoids GIL)
    """

    def __init__(
        self,
        vector_size: int,
        window: int,
        min_count: int,
        workers: int,
        model_path: str,
        normalize: bool = False,
    ):
        """
        Initialize the Word2VecEmbedding instance.

        Args:
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker processes for parallel embedding generation.
            model_path (str): Path to the pre-trained Word2Vec model file.
            normalize (bool): Whether to L2-normalize output embeddings. Default is False.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model_path = model_path
        self.normalize = normalize

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

        # Precompute the mean embedding for OOV fallback (better than zeros)
        self._mean_vector = self._compute_mean_vector()
        logger.info(
            f"Word2VecEmbedding initialized: vector_size={vector_size}, "
            f"vocab_size={len(self.model.key_to_index)}, normalize={normalize}"
        )

    def _compute_mean_vector(self) -> np.ndarray:
        """
        Compute the average embedding vector across the entire Word2Vec vocabulary.
        Used as the fallback for completely unknown tokens (better than zero vectors).

        Returns:
            np.ndarray: The mean vector of shape (vector_size,).
        """
        if len(self.model.key_to_index) > 0:
            return np.mean(self.model.vectors, axis=0).astype(np.float32)
        return np.zeros(self.vector_size, dtype=np.float32)

    def generate_embeddings(
        self,
        preprocessed_conversations: List[str],
        cache_key: str,
        batch_size: int = 1000,
    ) -> List[np.ndarray]:
        """
        Generate Word2Vec embeddings for the given preprocessed conversations
        using multiprocessing and batch processing.

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
        Generate Word2Vec embeddings in parallel using multiprocessing
        (ProcessPoolExecutor instead of ThreadPoolExecutor to avoid the GIL
        for CPU-bound embedding lookups).

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

        # Prepare worker arguments: each worker gets the model path and will
        # load its own copy (KeyedVectors is not picklable across processes).
        mean_vector_list = self._mean_vector.tolist()
        worker_args = [
            (batch, self.model_path, self.vector_size, self.normalize, mean_vector_list)
            for batch in batches
        ]

        # Use ProcessPoolExecutor for true parallelism on CPU-bound work
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_process_batch_worker, args): i
                for i, args in enumerate(
                    tqdm(worker_args, desc="Submitting tasks...")
                )
            }

            # Collect results preserving original order
            results = [None] * len(batches)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating word embedding batches...",
            ):
                batch_index = futures[future]
                try:
                    results[batch_index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index}: {str(e)}")
                    # Fill with mean vectors as fallback for failed batches
                    batch_len = len(batches[batch_index])
                    results[batch_index] = [
                        self._mean_vector.copy() for _ in range(batch_len)
                    ]

        # Flatten batch results into a single list
        embeddings = [embedding for batch in results for embedding in batch]
        return embeddings

    def _process_batch(self, batch: List[str]) -> List[np.ndarray]:
        """
        Process a batch of tokenized texts and generate embeddings in the main process.
        This is a fallback for single-process execution or debugging.

        Uses the multi-level OOV fallback strategy:
            1. Direct Word2Vec lookup
            2. Subword decomposition
            3. Character n-gram hashing
            4. Mean embedding fallback

        Args:
            batch (List[str]): Batch of tokenized texts.

        Returns:
            List[np.ndarray]: List of word embeddings for each text in the batch.
        """
        batch_embeddings = []
        for text in batch:
            if isinstance(text, str):
                embedding = _get_embedding_with_fallbacks(
                    text, self.model, self.vector_size, self._mean_vector
                )
            else:
                embedding = self._mean_vector.copy()

            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            batch_embeddings.append(embedding)

        return batch_embeddings
