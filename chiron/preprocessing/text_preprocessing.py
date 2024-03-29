import concurrent.futures
import random
import re
from collections import Counter
from typing import List, Dict, Any

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

    @staticmethod
    def count_token_frequencies(tokenized_texts: List[List[str]]) -> Counter:
        """
        Count token frequencies in the tokenized texts using ThreadPoolExecutor.

        Args:
            tokenized_texts (List[List[str]]): List of tokenized texts.

        Returns:
            Counter: Token frequency counts.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(lambda: Counter(text))
                for text in tqdm(tokenized_texts, desc="submitting tasks...")
            ]

            token_counts = Counter()
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Counting token frequencies",
            ):
                token_counts.update(future.result())

        return token_counts

    def build_vocabulary(self, tokenized_texts: List[List[str]]) -> None:
        """
        Build a vocabulary from the tokenized texts, handling special tokens and frequency thresholds.

        Args:
            tokenized_texts (List[List[str]]): List of tokenized texts.
        """
        token_counts = self.count_token_frequencies(tokenized_texts)

        self.vocab = {"<PAD>": 0, "<UNK>": 1}

        for token, count in tqdm(
            token_counts.most_common(self.max_vocab_size - 2),
            desc="Building vocabulary",
        ):  # noqa: E501
            if count >= self.min_freq:
                self.vocab[token] = len(self.vocab)

        logger.debug(f"vocab: {len(self.vocab)} entries")

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

    def augment_text(self, text: str) -> str:
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

    def preprocess_text_batch(self, batch: dict) -> List[List[str]]:
        preprocessed_batch = []
        # for batch in batches:
        human = ""
        ai = ""
        for dict_list in batch:
            for text_dict in dict_list:
                human = (
                    text_dict.get("value", "")
                    if text_dict.get("from") == "human"
                    else ""
                )
                ai = text_dict.get("value", "") if text_dict.get("from") == "ai" else ""

            final = f"\nhuman:\n{human}\nai:\n{ai}\n"
            preprocessed_text = self.preprocess_text(final)
            augmented_text = self.augment_text(
                preprocessed_text
            )  # Apply data augmentation
            tokenized_text = self.tokenize_text(augmented_text)
            preprocessed_batch.append(tokenized_text)

        return preprocessed_batch

    # chiron/preprocessing/text_preprocessing.py
    def preprocess(
        self, conversations: List[List[Dict[str, Any]]], cache_key: str
    ) -> List[List[str]]:
        preprocessed_data = load_cached_data(cache_key)
        if preprocessed_data is None:
            preprocessed_data = []
            for conversation in conversations:
                preprocessed_conversation = []
                for turn in conversation:
                    text = turn["value"]
                    preprocessed_text = self.preprocess_text(text)
                    preprocessed_conversation.append(preprocessed_text)
                preprocessed_data.append(preprocessed_conversation)
            cache_data(preprocessed_data, cache_key)
        return preprocessed_data

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess a single text by lowercasing, removing punctuation, and handling special characters.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Preprocessed text as a list of tokens.
        """
        preprocessed_text = self.preprocess_text_helper(text)
        augmented_text = self.augment_text(preprocessed_text)
        tokenized_text = self.tokenize_text(augmented_text)
        return tokenized_text

    def preprocess_text_helper(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d", "<NUM>", text)
        return text
