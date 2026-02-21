# pipeline.py

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer


class TextPredictionPipeline:
    """
    A pipeline for text prediction using an SNNModel or compatible model.

    Supports autoregressive generation via the model's forward() method,
    temperature-based sampling, beam search, batched inference, and
    embedding extraction mode (returning intermediate representations).

    Compatible with both HuggingFace generative models (via model.generate())
    and custom models like SNNModel that only expose forward().
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TextPredictionPipeline.

        Args:
            model: The model to use for inference. Can be a HuggingFace PreTrainedModel
                or any nn.Module (e.g., SNNModel) that implements forward().
            tokenizer: The tokenizer for encoding/decoding text.
            device: The device to run the model on.
            max_length: The maximum length of the generated text in tokens.
            num_return_sequences: The number of sequences to generate.
            **kwargs: Additional keyword arguments:
                - temperature (float): Sampling temperature. Default 1.0.
                    Values < 1.0 make output more deterministic.
                    Values > 1.0 make output more random.
                - top_k (int): Top-k sampling. Default 0 (disabled).
                - top_p (float): Nucleus (top-p) sampling. Default 1.0 (disabled).
                - beam_width (int): Beam search width. Default 0 (disabled, uses sampling).
                - adjacency_matrix: Adjacency matrix for SNNModel inference.
                - extract_embeddings (bool): If True, return intermediate representations
                    instead of decoded text. Default False.
                - repetition_penalty (float): Penalty for repeated tokens. Default 1.0.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.kwargs = kwargs

        # Generation parameters
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 0)
        self.top_p = kwargs.get("top_p", 1.0)
        self.beam_width = kwargs.get("beam_width", 0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        # SNNModel-specific parameters
        self.adjacency_matrix = kwargs.get("adjacency_matrix", None)

        # Embedding extraction mode
        self.extract_embeddings = kwargs.get("extract_embeddings", False)

    def _has_generate(self) -> bool:
        """Check whether the model has a generate() method (HuggingFace style)."""
        return hasattr(self.model, "generate") and callable(
            getattr(self.model, "generate")
        )

    def _prepare_adjacency_matrix(
        self, seq_len: int
    ) -> torch.Tensor:
        """
        Prepare the adjacency matrix for SNNModel forward pass.
        If no adjacency matrix is provided, creates an identity matrix as fallback.

        Args:
            seq_len: The sequence length to size the adjacency matrix.

        Returns:
            torch.Tensor: The adjacency matrix tensor on the correct device.
        """
        if self.adjacency_matrix is not None:
            if isinstance(self.adjacency_matrix, scipy.sparse.spmatrix):
                adj = torch.tensor(
                    self.adjacency_matrix.toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
            elif isinstance(self.adjacency_matrix, torch.Tensor):
                adj = self.adjacency_matrix.to(self.device)
            else:
                adj = torch.tensor(
                    self.adjacency_matrix,
                    dtype=torch.float32,
                    device=self.device,
                )
            return adj
        else:
            # Fallback: identity adjacency matrix (self-connections only)
            return torch.eye(seq_len, dtype=torch.float32, device=self.device)

    def _forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform a forward pass through the model, handling both SNNModel and
        HuggingFace model interfaces.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len).
            attention_mask: Attention mask tensor of shape (batch_size, seq_len).

        Returns:
            Tuple of (logits, embeddings).
            - logits: Output logits of shape (batch_size, output_size).
            - embeddings: Intermediate representations if available, else None.
        """
        batch_size, seq_len = input_ids.shape
        adj_matrix = self._prepare_adjacency_matrix(seq_len)

        # Create node indices (use index 0 for each batch element as default)
        node_indices = torch.zeros(
            batch_size, 1, dtype=torch.long, device=self.device
        )

        # Store intermediate outputs for embedding extraction
        embeddings = None

        if self.extract_embeddings and hasattr(self.model, "visualization_data"):
            # We will extract from visualization_data after forward
            output = self.model(input_ids, attention_mask, adj_matrix, node_indices)
            if self.model.visualization_data is not None:
                embeddings = {
                    "snn_output": self.model.visualization_data.get("snn_output"),
                    "gat_output": self.model.visualization_data.get("gat_output"),
                    "htm_output": self.model.visualization_data.get("htm_output"),
                    "final_output": self.model.visualization_data.get("final_output"),
                }
        else:
            output = self.model(input_ids, attention_mask, adj_matrix, node_indices)

        return output, embeddings

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, generated_ids: List[int]
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on previously generated tokens.

        Args:
            logits: Raw logits of shape (vocab_size,) or (output_size,).
            generated_ids: List of previously generated token IDs.

        Returns:
            torch.Tensor: Logits with repetition penalty applied.
        """
        if self.repetition_penalty == 1.0 or not generated_ids:
            return logits

        penalty = self.repetition_penalty
        for token_id in set(generated_ids):
            if token_id < logits.size(-1):
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty

        return logits

    def _sample_token(
        self, logits: torch.Tensor, generated_ids: List[int]
    ) -> int:
        """
        Sample a token from logits using temperature, top-k, and top-p filtering.

        Args:
            logits: Raw logits tensor of shape (vocab_size,).
            generated_ids: Previously generated token IDs for repetition penalty.

        Returns:
            int: The sampled token ID.
        """
        logits = self._apply_repetition_penalty(logits.clone(), generated_ids)

        # Apply temperature
        if self.temperature != 1.0 and self.temperature > 0:
            logits = logits / self.temperature

        # Apply top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[-1]
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Apply top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift so that the first token above threshold is kept
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()

        return token_id

    def _generate_greedy_or_sample(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Generate sequences using greedy/sampling decoding (one token at a time).

        Args:
            input_ids: Initial input token IDs of shape (1, seq_len).
            attention_mask: Attention mask of shape (1, seq_len).

        Returns:
            List[List[int]]: List of generated token ID sequences.
        """
        all_sequences = []

        for seq_idx in range(self.num_return_sequences):
            current_ids = input_ids.clone()
            current_mask = attention_mask.clone()
            generated_ids = []

            for step in range(self.max_length):
                with torch.no_grad():
                    output, _ = self._forward_pass(current_ids, current_mask)

                # output shape: (batch_size, output_size)
                # Take the output from the last position
                logits = output[0]  # (output_size,)

                # If output_size doesn't match vocab size, we treat it as a
                # projection and sample from available dimensions
                token_id = self._sample_token(logits, generated_ids)
                generated_ids.append(token_id)

                # Check for EOS token
                if (
                    hasattr(self.tokenizer, "eos_token_id")
                    and self.tokenizer.eos_token_id is not None
                    and token_id == self.tokenizer.eos_token_id
                ):
                    break

                # Update input for next step: append new token
                new_token = torch.tensor(
                    [[token_id]], dtype=torch.long, device=self.device
                )
                new_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)
                current_ids = torch.cat([current_ids, new_token], dim=1)
                current_mask = torch.cat([current_mask, new_mask], dim=1)

                # Truncate to max_length to prevent unbounded growth
                if current_ids.size(1) > self.max_length:
                    current_ids = current_ids[:, -self.max_length :]
                    current_mask = current_mask[:, -self.max_length :]

            all_sequences.append(generated_ids)

        return all_sequences

    def _generate_beam_search(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Generate sequences using beam search decoding.

        Args:
            input_ids: Initial input token IDs of shape (1, seq_len).
            attention_mask: Attention mask of shape (1, seq_len).

        Returns:
            List[List[int]]: Top-k beam sequences (token ID lists),
                where k = num_return_sequences.
        """
        beam_width = self.beam_width

        # Each beam: (log_probability, token_ids_list, current_input_ids, current_mask)
        initial_beam = (0.0, [], input_ids.clone(), attention_mask.clone())
        beams = [initial_beam]

        completed_beams = []

        for step in range(self.max_length):
            all_candidates = []

            for log_prob, generated_ids, curr_ids, curr_mask in beams:
                with torch.no_grad():
                    output, _ = self._forward_pass(curr_ids, curr_mask)

                logits = output[0]  # (output_size,)
                logits = self._apply_repetition_penalty(logits.clone(), generated_ids)

                # Apply temperature
                if self.temperature != 1.0 and self.temperature > 0:
                    logits = logits / self.temperature

                log_probs = F.log_softmax(logits, dim=-1)

                # Get top beam_width candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token_id = top_indices[i].item()
                    new_log_prob = log_prob + top_log_probs[i].item()
                    new_generated = generated_ids + [token_id]

                    # Check for EOS
                    if (
                        hasattr(self.tokenizer, "eos_token_id")
                        and self.tokenizer.eos_token_id is not None
                        and token_id == self.tokenizer.eos_token_id
                    ):
                        completed_beams.append((new_log_prob, new_generated))
                        continue

                    # Extend input
                    new_token = torch.tensor(
                        [[token_id]], dtype=torch.long, device=self.device
                    )
                    new_mask = torch.ones(
                        1, 1, dtype=torch.long, device=self.device
                    )
                    new_ids = torch.cat([curr_ids, new_token], dim=1)
                    new_attn = torch.cat([curr_mask, new_mask], dim=1)

                    # Truncate to max_length
                    if new_ids.size(1) > self.max_length:
                        new_ids = new_ids[:, -self.max_length :]
                        new_attn = new_attn[:, -self.max_length :]

                    all_candidates.append(
                        (new_log_prob, new_generated, new_ids, new_attn)
                    )

            if not all_candidates:
                break

            # Select top beam_width candidates by log probability
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

            # Early stop if we have enough completed beams
            if len(completed_beams) >= self.num_return_sequences:
                break

        # Add remaining beams to completed
        for log_prob, generated_ids, _, _ in beams:
            completed_beams.append((log_prob, generated_ids))

        # Sort by log probability (descending) and length-normalize
        completed_beams.sort(
            key=lambda x: x[0] / max(len(x[1]), 1), reverse=True
        )

        # Return top num_return_sequences
        return [
            seq for _, seq in completed_beams[: self.num_return_sequences]
        ]

    def _generate_with_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Generate sequences using the model's forward() method.
        Dispatches to beam search or sampling based on configuration.

        Args:
            input_ids: Input token IDs of shape (1, seq_len).
            attention_mask: Attention mask of shape (1, seq_len).

        Returns:
            List[List[int]]: Generated token ID sequences.
        """
        if self.beam_width > 1:
            return self._generate_beam_search(input_ids, attention_mask)
        else:
            return self._generate_greedy_or_sample(input_ids, attention_mask)

    def _extract_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate representations from the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of intermediate layer outputs.
        """
        with torch.no_grad():
            output, embeddings = self._forward_pass(input_ids, attention_mask)

        if embeddings is not None:
            return embeddings

        # Fallback: return the final output as the only embedding
        return {"final_output": output}

    def __call__(self, text: str) -> List[str]:
        """
        Generate text based on the input text, or extract embeddings if
        extract_embeddings mode is enabled.

        Handles both HuggingFace models (with generate()) and SNNModel-style
        models (forward() only) transparently.

        Args:
            text (str): The input text to generate from.

        Returns:
            List[str]: The list of generated texts. If extract_embeddings is True,
                returns a list containing a string representation of the embedding
                dictionary (for interface compatibility).
        """
        # Tokenize the input text
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Embedding extraction mode
        if self.extract_embeddings:
            embeddings = self._extract_embeddings(input_ids, attention_mask)
            # Return string representations for interface compatibility
            embedding_strs = []
            for name, tensor in embeddings.items():
                if tensor is not None:
                    embedding_strs.append(
                        f"{name}: shape={list(tensor.shape)}"
                    )
            return embedding_strs if embedding_strs else ["No embeddings extracted"]

        # Generation mode
        self.model.eval()

        if self._has_generate():
            # HuggingFace model with generate() method
            with torch.no_grad():
                # Filter kwargs to only pass supported generation parameters
                gen_kwargs = {}
                for key in ("do_sample", "top_k", "top_p", "repetition_penalty",
                            "no_repeat_ngram_size", "early_stopping"):
                    if key in self.kwargs:
                        gen_kwargs[key] = self.kwargs[key]

                if self.temperature != 1.0:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs.setdefault("do_sample", True)

                if self.beam_width > 1:
                    gen_kwargs["num_beams"] = self.beam_width

                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_return_sequences=self.num_return_sequences,
                    **gen_kwargs,
                )
            generated_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
        else:
            # SNNModel or other custom model: use forward()-based generation
            generated_sequences = self._generate_with_forward(
                input_ids, attention_mask
            )
            generated_texts = []
            for seq in generated_sequences:
                # Filter out invalid token IDs before decoding
                vocab_size = self.tokenizer.vocab_size
                valid_ids = [
                    tid for tid in seq
                    if 0 <= tid < vocab_size
                ]
                if valid_ids:
                    decoded = self.tokenizer.decode(
                        valid_ids, skip_special_tokens=True
                    )
                else:
                    decoded = ""
                generated_texts.append(decoded)

        return generated_texts

    def generate_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Perform batched inference on multiple input texts.

        For models with generate(), uses native batched generation.
        For SNNModel-style models, processes each text through forward()-based
        generation and collects results.

        Args:
            texts: List of input texts.

        Returns:
            List[List[str]]: List of generated text lists, one per input text.
        """
        if not texts:
            return []

        self.model.eval()

        if self._has_generate():
            # Batch encode for HuggingFace models
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            gen_kwargs = {}
            if self.temperature != 1.0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            if self.beam_width > 1:
                gen_kwargs["num_beams"] = self.beam_width

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_return_sequences=self.num_return_sequences,
                    **gen_kwargs,
                )

            batch_size = len(texts)
            all_results = []
            for i in range(batch_size):
                start = i * self.num_return_sequences
                end = start + self.num_return_sequences
                batch_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in output_ids[start:end]
                ]
                all_results.append(batch_texts)

            return all_results
        else:
            # For forward()-based models, process each text individually
            # (true batching requires model-specific changes)
            results = []
            for text in texts:
                generated = self(text)
                results.append(generated)
            return results

    def extract_embeddings_batch(
        self, texts: List[str]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract embeddings for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List[Dict[str, torch.Tensor]]: Embedding dictionaries for each input.
        """
        results = []
        old_extract = self.extract_embeddings
        self.extract_embeddings = True

        try:
            for text in texts:
                encoded = self.tokenizer.encode_plus(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                embeddings = self._extract_embeddings(input_ids, attention_mask)
                results.append(embeddings)
        finally:
            self.extract_embeddings = old_extract

        return results
