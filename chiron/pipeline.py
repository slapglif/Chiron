# pipeline.py
from typing import Any, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class TextPredictionPipeline:
    """
    A pipeline for text prediction using a pre-trained language model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TextPredictionPipeline.

        Args:
            model (PreTrainedModel): The pre-trained language model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
            device (torch.device): The device to run the model on.
            max_length (int): The maximum length of the generated text.
            num_return_sequences (int): The number of sequences to generate.
            **kwargs (Any): Additional keyword arguments for text generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.kwargs = kwargs

    def __call__(self, text: str) -> List[str]:
        """
        Generate text based on the input text.

        Args:
            text (str): The input text to generate from.

        Returns:
            List[str]: The list of generated texts.
        """
        # Tokenize the input text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        # Generate text
        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_return_sequences=self.num_return_sequences,
            **self.kwargs,
        )

        # Decode the generated text
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]

        return generated_texts
