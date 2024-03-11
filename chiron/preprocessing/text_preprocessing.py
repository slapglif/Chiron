import concurrent.futures
import re
from collections import Counter
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

from chiron.utils.cache import load_cached_data, cache_data


class TextPreprocessor:
    """Class for preprocessing text data."""

    def __init__(self, min_freq: int, max_vocab_size: int):
        """
        Initialize the TextPreprocessor.

        Args:
            min_freq (int): Minimum frequency threshold for including a token in the vocabulary.
            max_vocab_size (int): Maximum size of the vocabulary.
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = None

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the input text by lowercasing, removing punctuation, and handling special characters.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d", "<NUM>", text)
        return text

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

    def preprocess_text_batch(self, batch: dict) -> List[List[str]]:
        preprocessed_batch = []
        # for batch in batches:
        human = ""
        ai = ""
        for dict_list in batch:
            for text_dict in dict_list:
                human = text_dict.get("value", "") if text_dict.get("from") == "human" else ""
                ai = text_dict.get("value", "") if text_dict.get("from") == "ai" else ""

            final = f"\nhuman:\n{human}\nai:\n{ai}\n"
            preprocessed_text = self.preprocess_text(final)
            tokenized_text = self.tokenize_text(preprocessed_text)
            preprocessed_batch.append(tokenized_text)

        return preprocessed_batch

    def preprocess(
            self, texts: Dict[str, Any], cache_key: str
    ) -> List[List[str]]:
        preprocessed_data = load_cached_data(cache_key)
        if preprocessed_data is None:
            preprocessed_data = self.preprocess_text_batch(texts)
            cache_data(preprocessed_data, cache_key)
        return preprocessed_data
