# text_preprocessing.py

import concurrent.futures
import functools
import math
import multiprocessing
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from loguru import logger
from nltk.corpus import wordnet
from tqdm import tqdm

from chiron.utils.cache import cache_data, load_cached_data

# Pre-compiled regex patterns for tokenization
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+"
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"
)
_CONTRACTION_PATTERN = re.compile(
    r"(?i)\b(can't|won't|shouldn't|wouldn't|couldn't|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|hasn't|haven't|hadn't|mustn't|needn't|shan't|"
    r"i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|"
    r"i'd|you'd|he'd|she'd|we'd|they'd|i'll|you'll|he'll|she'll|we'll|they'll|"
    r"that's|who's|what's|here's|there's|where's|when's|why's|how's|"
    r"let's|who've|who'll|who'd|what'll|what're|what've|"
    r"ain't|o'clock|ma'am|y'all|'twas)\b"
)
# Tokenizer pattern: URLs, emails, contractions, words (with hyphens), punctuation, numbers
_TOKEN_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+)"           # URLs
    r"|(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})"  # Emails
    r"|(?:(?i)(?:can't|won't|shouldn't|wouldn't|couldn't|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|hasn't|haven't|hadn't|mustn't|needn't|"
    r"i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|"
    r"i'd|you'd|he'd|she'd|we'd|they'd|i'll|you'll|he'll|she'll|we'll|they'll|"
    r"that's|who's|what's|here's|there's|where's|when's|why's|how's|"
    r"let's|ain't|o'clock|ma'am|y'all))"  # Contractions
    r"|(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"  # Numbers (int, float, scientific)
    r"|(?:[A-Za-z]+(?:[-'][A-Za-z]+)*)"     # Words (with hyphens and apostrophes)
    r"|(?:[.,!?;:\"'()\[\]{}\-/\\@#$%^&*+=<>~`|])"  # Individual punctuation
)


def _get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word from WordNet."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.append(candidate)
    return list(set(synonyms))


def _process_single_conversation(
    args: Tuple[List[Dict[str, Any]], float],
) -> List[str]:
    """
    Process a single conversation in a worker process.
    This is a module-level function so it can be pickled for multiprocessing.

    Args:
        args: Tuple of (conversation dict_list, augmentation_prob).

    Returns:
        List[str]: Preprocessed and augmented text strings from this conversation.
    """
    dict_list, augmentation_prob = args
    results = []

    human = ""
    ai = ""
    for text_dict in dict_list:
        if isinstance(text_dict, dict):
            from_field = text_dict.get("from", "")
            value = text_dict.get("value", "")
            if from_field == "human":
                human = value
            elif from_field in ("ai", "gpt"):
                ai = value

    final = f"\nhuman:\n{human}\nai:\n{ai}\n"
    preprocessed_text = _preprocess_text_helper_static(final)
    augmented_text = _augment_text_static(preprocessed_text, augmentation_prob)
    tokenized_text = _tokenize_text_static(augmented_text)
    results.append(" ".join(tokenized_text))

    return results


def _preprocess_text_helper_static(text: str) -> str:
    """
    Preprocess a single text: lowercase, normalize whitespace, replace numbers.
    BPE-aware: preserves punctuation since subword tokenizers use it.
    """
    text = text.lower()
    # Replace standalone numbers with <NUM> but preserve punctuation for BPE
    text = _NUMBER_PATTERN.sub("<NUM>", text)
    # Normalize whitespace (collapse multiple spaces/tabs/newlines except for structural newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize_text_static(text: str) -> List[str]:
    """
    Tokenize text using regex-based tokenization that handles contractions,
    URLs, numbers, emails, and punctuation properly.
    """
    tokens = _TOKEN_PATTERN.findall(text)
    return tokens


def _augment_text_static(text: str, augmentation_prob: float = 0.1) -> str:
    """
    Augment text using EDA (Easy Data Augmentation) techniques:
    1. Synonym replacement (SR)
    2. Random insertion (RI)
    3. Random swap (RS)
    4. Random deletion (RD)

    Each technique is applied with equal probability when augmentation triggers.
    """
    words = text.split()
    if len(words) < 2:
        return text

    # Number of words to modify (at least 1)
    n_augment = max(1, int(len(words) * augmentation_prob))

    # Choose a random EDA technique
    technique = random.choice(["synonym_replace", "random_insert", "random_swap", "random_delete"])

    if technique == "synonym_replace":
        words = _synonym_replace(words, n_augment)
    elif technique == "random_insert":
        words = _random_insert(words, n_augment)
    elif technique == "random_swap":
        words = _random_swap(words, n_augment)
    elif technique == "random_delete":
        words = _random_delete(words, augmentation_prob)

    return " ".join(words)


def _synonym_replace(words: List[str], n: int) -> List[str]:
    """Replace n random words with their synonyms."""
    new_words = words.copy()
    # Only consider words that are alphabetic (skip punctuation, numbers, special tokens)
    candidate_indices = [
        i for i, w in enumerate(new_words)
        if w.isalpha() and not w.startswith("<") and not w.endswith(">")
    ]
    random.shuffle(candidate_indices)

    replacements_made = 0
    for idx in candidate_indices:
        if replacements_made >= n:
            break
        synonyms = _get_synonyms(new_words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            replacements_made += 1

    return new_words


def _random_insert(words: List[str], n: int) -> List[str]:
    """Insert n random synonyms of random words at random positions."""
    new_words = words.copy()
    for _ in range(n):
        # Pick a random word that is alphabetic
        alpha_words = [w for w in new_words if w.isalpha() and not w.startswith("<")]
        if not alpha_words:
            break
        random_word = random.choice(alpha_words)
        synonyms = _get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, synonym)
    return new_words


def _random_swap(words: List[str], n: int) -> List[str]:
    """Randomly swap n pairs of words."""
    new_words = words.copy()
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def _random_delete(words: List[str], p: float) -> List[str]:
    """Randomly delete words with probability p. Always keep at least one word."""
    if len(words) <= 1:
        return words
    new_words = [w for w in words if random.random() > p]
    # Ensure at least one word remains
    if not new_words:
        return [random.choice(words)]
    return new_words


class TextPreprocessor:
    def __init__(self, min_freq: int, max_vocab_size: int, **kwargs):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.config = kwargs
        # Configurable augmentation probability (default 0.1 = 10%)
        self.augmentation_prob = kwargs.get("augmentation_prob", 0.1)
        # Number of worker processes for parallel preprocessing
        self.num_workers = kwargs.get("num_workers", None)  # None = use cpu_count

    def tokens_to_indices(self, tokenized_texts: List[str]) -> List[int]:
        """
        Convert tokenized text to numerical indices based on the vocabulary.

        Args:
            tokenized_texts (List[str]): The tokenized text.

        Returns:
            List[int]: A list of numerical indices representing the tokenized text.
        """
        if self.vocab is None:
            raise ValueError("Vocabulary has not been built yet.")

        return [
            self.vocab.get(token, self.vocab.get("<UNK>", 0))
            for token in tokenized_texts
        ]

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenize the given text using regex-based tokenization that handles
        contractions, URLs, numbers, emails, and punctuation.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return _tokenize_text_static(text)

    @classmethod
    def count_token_frequencies(cls, tokenized_texts: List[str]) -> Counter:
        """
        Count token frequencies in the tokenized texts using ThreadPoolExecutor.

        Args:
            tokenized_texts (List[str]): List of tokenized texts.

        Returns:
            Counter: Token frequency counts.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(lambda tokens=tokens: Counter(tokens))
                for tokens in tqdm(tokenized_texts, desc="Submitting tasks...")
            ]

            token_counts = Counter()
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Counting token frequencies",
            ):
                token_counts.update(future.result())

        return token_counts

    def build_vocabulary(self, tokenized_texts: List[str]) -> None:
        """
        Build a vocabulary from the tokenized texts, handling special tokens and frequency thresholds.

        Args:
            tokenized_texts (List[str]): List of tokenized texts.
        """
        token_counts = self.count_token_frequencies(tokenized_texts)

        self.vocab = {"<PAD>": 0, "<UNK>": 1}

        for token, count in tqdm(
            token_counts.most_common(self.max_vocab_size - 2),
            desc="Building vocabulary",
        ):
            if count >= self.min_freq:
                self.vocab[token] = len(self.vocab)

        logger.debug(f"Vocabulary size: {len(self.vocab)} entries")

    @staticmethod
    def validate_batch(batch: List[Dict[str, Any]]) -> bool:
        """
        Validate that each item in the batch is a dictionary and has the expected keys.
        Checks for 'gpt' in addition to 'ai' and 'human' as valid 'from' field values.

        Args:
            batch (List[Dict[str, Any]]): The batch to validate.

        Returns:
            bool: True if the batch is valid, False otherwise.
        """
        valid_from_values = {"human", "ai", "gpt"}
        return all(
            isinstance(item, dict)
            and "from" in item
            and "value" in item
            and item.get("from") in valid_from_values
            for item in batch
        )

    @classmethod
    def augment_text(cls, text: str) -> str:
        """
        Augment the input text by applying EDA (Easy Data Augmentation) techniques:
        synonym replacement, random insertion, random swap, and random deletion.

        Args:
            text (str): Input text.

        Returns:
            str: Augmented text.
        """
        return _augment_text_static(text, augmentation_prob=0.1)

    def preprocess_text_batch(self, batch: List[List[Dict[str, Any]]]) -> List[str]:
        preprocessed_batch = []
        for dict_list in batch:
            human = ""
            ai = ""
            for text_dict in dict_list:
                if isinstance(text_dict, dict):
                    from_field = text_dict.get("from", "")
                    value = text_dict.get("value", "")
                    if from_field == "human":
                        human = value
                    elif from_field in ("ai", "gpt"):
                        ai = value

            final = f"\nhuman:\n{human}\nai:\n{ai}\n"
            preprocessed_text = self.preprocess_text_helper(final)
            augmented_text = _augment_text_static(preprocessed_text, self.augmentation_prob)
            tokenized_text = self.tokenize_text(augmented_text)
            preprocessed_batch.append(" ".join(tokenized_text))

        return preprocessed_batch

    def preprocess_text_helper(self, text: str) -> str:
        """
        Preprocess a single text by lowercasing, handling numbers, and normalizing whitespace.
        BPE-aware: preserves punctuation since subword tokenizers rely on it.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        return _preprocess_text_helper_static(text)

    def preprocess(
        self, conversations: List[List[Dict[str, Any]]], cache_key: str
    ) -> List[str]:
        """
        Preprocess conversations using multiprocessing for parallelism.

        Args:
            conversations: List of conversations, each a list of dicts with 'from' and 'value'.
            cache_key: Key for caching preprocessed results.

        Returns:
            List[str]: List of preprocessed text strings.
        """
        preprocessed_data = load_cached_data(cache_key)
        if preprocessed_data is not None:
            return preprocessed_data

        # Filter valid conversations first
        valid_conversations = [
            conv for conv in conversations if self.validate_batch(conv)
        ]

        if not valid_conversations:
            logger.warning("No valid conversations found to preprocess.")
            preprocessed_data = []
            cache_data(preprocessed_data, cache_key)
            return preprocessed_data

        # Determine number of workers
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(valid_conversations))
        num_workers = max(1, min(num_workers, len(valid_conversations)))

        # Prepare arguments for each worker: (conversation, augmentation_prob)
        worker_args = [
            (conv, self.augmentation_prob) for conv in valid_conversations
        ]

        preprocessed_data = []

        if num_workers <= 1 or len(valid_conversations) < 10:
            # Fall back to sequential for small datasets or single worker
            for args in tqdm(worker_args, desc="Preprocessing conversations"):
                result = _process_single_conversation(args)
                preprocessed_data.extend(result)
        else:
            # Use multiprocessing Pool for parallel preprocessing
            logger.info(
                f"Preprocessing {len(valid_conversations)} conversations "
                f"with {num_workers} workers"
            )
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_process_single_conversation, worker_args, chunksize=64),
                        total=len(worker_args),
                        desc="Preprocessing conversations (parallel)",
                    )
                )
            for result in results:
                preprocessed_data.extend(result)

        cache_data(preprocessed_data, cache_key)
        return preprocessed_data

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
