# text_preprocessing.py

import concurrent.futures
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import numpy as np
from loguru import logger
from nltk.corpus import wordnet
from tqdm import tqdm

from chiron.utils.cache import cache_data, load_cached_data


class TextPreprocessor:
    def __init__(self, min_freq: int, max_vocab_size: int, **kwargs):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.config = kwargs

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
        Tokenize the given text into individual tokens using whitespace splitting.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return text.split()

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

        Args:
            batch (List[Dict[str, Any]]): The batch to validate.

        Returns:
            bool: True if the batch is valid, False otherwise.
        """
        return all(
            isinstance(item, dict) and "from" in item and "value" in item
            for item in batch
        )

    @classmethod
    def augment_text(cls, text: str) -> str:
        """
        Augment the input text by applying random transformations.

        Args:
            text (str): Input text.

        Returns:
            str: Augmented text.
        """
        words = text.split()
        augmented_words = []

        for word in words:
            if random.random() < 0.1:  # 10% chance of augmentation
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())

                if synonyms:
                    augmented_word = random.choice(synonyms)
                else:
                    augmented_word = word
            else:
                augmented_word = word

            augmented_words.append(augmented_word)

        augmented_text = " ".join(augmented_words)
        return augmented_text

    def preprocess_text_batch(self, batch: List[List[Dict[str, Any]]]) -> List[str]:
        preprocessed_batch = []
        for dict_list in batch:
            human = ""
            ai = ""
            for text_dict in dict_list:
                if isinstance(text_dict, dict):
                    human = (
                        text_dict.get("value", "")
                        if text_dict.get("from") == "human"
                        else human
                    )
                    ai = (
                        text_dict.get("value", "")
                        if text_dict.get("from") == "ai"
                        else ai
                    )

            final = f"\nhuman:\n{human}\nai:\n{ai}\n"
            preprocessed_text = self.preprocess_text_helper(final)
            augmented_text = self.augment_text(preprocessed_text)
            tokenized_text = self.tokenize_text(augmented_text)
            preprocessed_batch.append(" ".join(tokenized_text))

        return preprocessed_batch

    def preprocess_text_helper(self, text: str) -> str:
        """
        Preprocess a single text by lowercasing, removing punctuation, and handling special characters.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d", "<NUM>", text)
        return text

    def preprocess(
        self, conversations: List[List[Dict[str, Any]]], cache_key: str
    ) -> List[str]:
        preprocessed_data = load_cached_data(cache_key)
        if preprocessed_data is None:
            preprocessed_data = []
            for conversation in conversations:
                if self.validate_batch(conversation):
                    preprocessed_conversation = self.preprocess_text_batch(conversation)
                    preprocessed_data.extend(preprocessed_conversation)
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
