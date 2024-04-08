# chiron/utils/tokenization.py
import numpy as np
from transformers import BertTokenizer


class EmbeddingTokenizer:
    def __init__(self, vocab_size: int, pad_token: str = "[PAD]", unk_token: str = "[UNK]"):
        """
        Initialize the EmbeddingTokenizer.

        Args:
            vocab_size (int): The size of the vocabulary.
            pad_token (str): The padding token.
            unk_token (str): The unknown token.
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token_id = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"pad_token": pad_token, "unk_token": unk_token})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode the binarized embeddings into token IDs.

        Args:
            embeddings (np.ndarray): The binarized embeddings.

        Returns:
            np.ndarray: The token IDs.
        """
        # Convert binarized embeddings to string format
        embeddings_str = [" ".join(map(str, emb)) for emb in embeddings]

        # Tokenize the string embeddings
        token_ids = self.tokenizer.batch_encode_plus(
            embeddings_str,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="np",
        )["input_ids"]

        return token_ids

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Decode the token IDs back to binarized embeddings.

        Args:
            token_ids (np.ndarray): The token IDs.

        Returns:
            np.ndarray: The binarized embeddings.
        """
        # Decode the token IDs to string format
        embeddings_str = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        # Convert string embeddings back to binarized format
        embeddings = []
        for emb_str in embeddings_str:
            emb = list(map(int, emb_str.split()))
            embeddings.append(emb)

        return np.array(embeddings)
