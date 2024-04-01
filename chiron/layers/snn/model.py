# chiron/layers/snn/model.py
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from chiron.layers.htm.model import HTMModel
from chiron.layers.snn.graph_attention import GraphAttentionLayer


def batch_cosine_similarity(
    sdr_embeddings: torch.Tensor, threshold: float = 0.5, batch_size: int = 1000
) -> torch.Tensor:
    """
    Computes cosine similarity in batches and directly produces a sparse matrix to avoid memory overflow.

    Args:
        sdr_embeddings (torch.Tensor): SDR embeddings of shape (N, D) where N is the number of embeddings and D is the dimension.
        threshold (float): Cosine similarity threshold to consider two vectors as connected.
        batch_size (int): The size of each batch for computation.

    Returns:
        torch.sparse.Tensor: A sparse matrix of cosine similarities above the threshold.
    """
    num_embeddings = sdr_embeddings.size(0)
    device = sdr_embeddings.device

    indices = []
    values = []

    for i in tqdm(
        range(0, num_embeddings, batch_size), desc="Computing cosine similarity"
    ):
        batch_end = min(i + batch_size, num_embeddings)
        batch = sdr_embeddings[i:batch_end]

        # Efficient computation of cosine similarity using matrix multiplication
        similarity = torch.mm(batch, sdr_embeddings.t())

        # Convert to sparse format immediately to save memory
        batch_indices = torch.nonzero(similarity > threshold, as_tuple=False).t()
        batch_values = similarity[similarity > threshold]

        # Adjust indices for the current batch
        batch_indices[0] += i

        indices.append(batch_indices)
        values.append(batch_values)

    # Concatenate all indices and values
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values, dim=0)

    # Create the final sparse matrix
    similarity_matrix_sparse = torch.sparse_coo_tensor(
        indices,
        values,
        (num_embeddings, num_embeddings),
        device=device,
    )

    return similarity_matrix_sparse


def create_adjacency_matrix(
    sdr_embeddings: torch.Tensor, threshold: float = 0.5
) -> torch.sparse.Tensor:
    """
    Generates an adjacency matrix from SDR embeddings using cosine similarity.

    Args:
        sdr_embeddings (torch.Tensor): The SDR embeddings tensor of shape (N, D).
        threshold (float): Threshold for considering two embeddings as neighbors.

    Returns:
        torch.sparse.Tensor: The adjacency matrix as a sparse COO tensor.
    """
    # Normalize embeddings to unit vectors for cosine similarity calculation
    sdr_embeddings = sdr_embeddings / sdr_embeddings.norm(dim=1, keepdim=True)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(sdr_embeddings, sdr_embeddings.t())

    # Apply threshold to create a binary adjacency matrix
    adjacency_matrix = (similarity_matrix > threshold).float()

    # Convert the adjacency matrix to a sparse COO tensor
    indices = torch.nonzero(adjacency_matrix).t()
    values = adjacency_matrix[indices[0], indices[1]]
    adjacency_matrix_sparse = torch.sparse_coo_tensor(
        indices, values, adjacency_matrix.size()
    )

    return adjacency_matrix_sparse


def create_adjacency_matrix_batched(
    sdr_embeddings: torch.Tensor, threshold: float = 0.5, batch_size: int = 10000
) -> torch.sparse.Tensor:
    num_embeddings = sdr_embeddings.size(0)
    indices = []
    values = []

    # Normalize embeddings to unit vectors for cosine similarity calculation
    sdr_embeddings = sdr_embeddings / sdr_embeddings.norm(dim=1, keepdim=True)

    num_batches = (num_embeddings + batch_size - 1) // batch_size
    progress_bar = tqdm(total=num_batches, desc="Computing adjacency matrix")

    for i in range(0, num_embeddings, batch_size):
        batch_end = min(i + batch_size, num_embeddings)
        batch_embeddings = sdr_embeddings[i:batch_end]

        # Compute cosine similarity for the current batch
        similarity_matrix = torch.einsum("nd,md->nm", batch_embeddings, sdr_embeddings)

        # Apply threshold and convert to sparse format
        batch_indices = torch.nonzero(similarity_matrix > threshold).t()
        batch_values = similarity_matrix[batch_indices[0], batch_indices[1]]

        indices.append(batch_indices)
        values.append(batch_values)

        progress_bar.update(1)

    progress_bar.close()

    # Concatenate indices and values from all batches
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)

    adjacency_matrix_sparse = torch.sparse_coo_tensor(
        indices, values, (num_embeddings, num_embeddings)
    )

    return adjacency_matrix_sparse


class SNNLayer(nn.Module):
    """
    Spiking Neural Network (SNN) layer.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        timesteps: int,
        num_nodes: int,
        dropout: float = 0.0,
    ):
        super(SNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.timesteps = timesteps
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.fc1 = None  # will be dynamically defined based on input size
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1 is None:
            input_size = x.size(-1)
            self.fc1 = nn.Linear(input_size, self.hidden_size).to(x.device)

        # Ensure the input tensor is of the same type as the model's parameters
        x = x.to(dtype=self.fc2.weight.dtype, device=x.device)

        if (
            x.dim() == 2
        ):  # If input is 2D (batch_size, feature_size), add a sequence dimension
            x = x.unsqueeze(1)  # Reshape to (batch_size, 1, feature_size)

        batch_size, seq_len, _ = x.shape

        mem1 = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype
        )
        mem2 = torch.zeros(
            batch_size, seq_len, self.output_size, device=x.device, dtype=x.dtype
        )

        spikes = []
        for _ in range(self.timesteps):
            mem1 += self.fc1(x)
            spike1 = (mem1 > 0.5).float()
            mem1 *= mem1 <= 0.5

            spike1 = self.dropout_layer(spike1)
            mem2 += self.fc2(spike1)
            spike2 = (mem2 > 0.5).float()
            mem2 *= mem2 <= 0.5

            spikes.append(spike2)

        output = torch.stack(spikes, dim=1).mean(dim=1)

        return output


class SNNModel(nn.Module):
    """
    Spiking Neural Network (SNN) model.
    Args:
        sp_params (dict): Parameters for the spatial pooling layer.
        gat_params (dict): Parameters for the Graph Attention layer.
        htm_params (dict): Parameters for the Hierarchical Temporal Memory layer.
        device (torch.device): The device to run the model on.
        vocab (dict): The vocabulary mapping for token to index.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing text.
        snn_params (dict): Parameters for the SNN layer.
    """

    def __init__(
        self,
        sp_params: dict,
        gat_params: dict,
        htm_params: dict,
        device: torch.device,
        vocab: dict,
        tokenizer: PreTrainedTokenizer,
        snn_params: dict,
    ):
        super(SNNModel, self).__init__()
        self.sp_params = sp_params
        self.gat_params = gat_params
        self.htm_params = htm_params
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.output_size = sp_params["sdr_dimensions"]

        self.snn_layer = SNNLayer(
            hidden_size=snn_params["hidden_size"],
            output_size=snn_params["output_size"],
            timesteps=snn_params["timesteps"],
            num_nodes=self.output_size,
            dropout=snn_params.get("dropout", 0.0),
        ).to(device)

        self.gat_layer = GraphAttentionLayer(
            in_features=gat_params["in_features"],
            out_features=gat_params["out_features"],
            num_heads=gat_params["num_heads"],
            alpha=gat_params["alpha"],
        ).to(device)

        # Ensure the output size of SNNLayer matches the expected input size of HTMModel
        self.htm_layer = HTMModel(
            sdr_dimensions=self.output_size, device=device, **htm_params
        ).to(device)

        self.fc_out = nn.Linear(self.output_size, self.output_size).to(device)

        logger.info(
            f"SNNModel initialized with sp_params={sp_params}, gat_params={gat_params},"
            f"htm_params={htm_params}, output_size={self.output_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrix: torch.sparse_coo_tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the SNN model.
        Args:
            input_ids (torch.Tensor): The input token IDs tensor of shape (batch_size, seq_len) or (batch_size,).
            attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, seq_len) or (batch_size,).
            adj_matrix (torch.sparse_coo_tensor): The adjacency matrix in sparse COO format.
            node_indices (torch.Tensor): The node indices tensor.
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        logger.debug(f"Input tensor shape: {input_ids.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")
        logger.debug(f"Adjacency matrix shape: {adj_matrix.shape}")
        logger.debug(f"Node indices shape: {node_indices.shape}")

        # Add an extra dimension if input_ids is 1D
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        batch_size, seq_len = input_ids.size()

        # Move the input tensors to the specified device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Apply the SNN layer
        snn_output = self.snn_layer(input_ids)
        logger.debug(f"SNN output shape: {snn_output.shape}")

        # Apply the GAT layer
        gat_output = self.gat_layer(snn_output, adj_matrix)
        logger.debug(f"GAT output shape: {gat_output.shape}")

        # Apply the HTM layer
        htm_output = self.htm_layer(gat_output).view(batch_size, -1)
        logger.debug(f"HTM output shape: {htm_output.shape}")

        # Apply the final fully connected layer
        output = self.fc_out(htm_output)
        logger.debug(f"Final output shape: {output.shape}")

        return output

    def generate(
            self,
            input_conversation: list,
            adjacency_matrix: torch.Tensor,
            node_indices: torch.Tensor,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            mirostat_eta: float = 0.1,
            mirostat_tau: float = 5.0,
            **kwargs,
    ) -> list:
        """
        Generate responses based on the input conversation.

        Args:
            input_conversation (list): The input conversation as a list of strings or tensors.
            adjacency_matrix (torch.Tensor): The adjacency matrix tensor.
            node_indices (torch.Tensor): The node indices tensor.
            max_length (int): The maximum length of the generated response.
            num_return_sequences (int): The number of responses to generate.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top-k tokens to consider for filtering.
            top_p (float): The cumulative probability threshold for filtering.
            mirostat_eta (float): The learning rate for Mirostat.
            mirostat_tau (float): The target entropy for Mirostat.

        Returns:
            list: The generated conversations as a list of strings.
        """
        generated_conversations = []

        # Convert the input conversation to a list of strings
        string_conversation = [
            str(turn.item()) if isinstance(turn, torch.Tensor) else str(turn)
            for turn in input_conversation
        ]

        # Check if the input conversation is empty
        if not string_conversation:
            logger.warning("Input conversation is empty. Skipping generation.")
            return generated_conversations

        for _ in range(num_return_sequences):
            generated_conversation = []

            # Concatenate the input conversation to start the generation
            input_text = " ".join(string_conversation)
            generated_conversation.append(input_text)

            # Initialize Mirostat parameters
            mirostat_mu = 2.0 * mirostat_tau
            mirostat_s = 1.0

            # Generate the model's response
            model_input = self._prepare_input(generated_conversation)
            input_ids = model_input["input_ids"].to(self.device)
            attention_mask = model_input["attention_mask"].to(self.device)

            response = []
            while len(response) < max_length:
                # Forward pass
                if adjacency_matrix is None or node_indices is None:
                    # If adjacency_matrix or node_indices are not provided, create dummy values
                    dummy_adj_matrix = torch.eye(input_ids.size(-1), dtype=torch.bool).to(self.device)
                    dummy_node_indices = torch.arange(input_ids.size(-1), device=self.device)
                    output = self.forward(input_ids, attention_mask, dummy_adj_matrix, dummy_node_indices)
                else:
                    output = self.forward(input_ids, attention_mask, adjacency_matrix, node_indices)

                # Sample from the model's output distribution
                output_logits = output[:, -1, :] / temperature
                filtered_logits = self._top_k_top_p_filtering(
                    output_logits, top_k=top_k, top_p=top_p
                )
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Update Mirostat parameters
                mirostat_s = (
                        mirostat_eta * (output_logits.max().item() - mirostat_tau)
                        + (1 - mirostat_eta) * mirostat_s
                )
                mirostat_mu = mirostat_mu * torch.exp(mirostat_s)
                temperature = max(0.1, temperature * (mirostat_tau / mirostat_mu))

                # Check if the generated token is the end-of-sequence token
                if next_token.item() == self.eos_token_id:
                    break

                response.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.tensor([1], device=self.device, dtype=torch.long)], dim=-1
                )

            # Add the generated response to the conversation
            generated_response = self.tokenizer.decode(response)
            generated_conversation.append(generated_response)
            generated_conversations.append(generated_conversation)

        return generated_conversations

    def _prepare_input(self, conversation: list) -> dict:
        """
        Prepare the input for the model based on the conversation structure.

        Args:
            conversation (list): The conversation as a list of strings.

        Returns:
            dict: The prepared input for the model.
        """
        input_ids = []

        for turn in conversation:
            if isinstance(turn, str):
                encoded_turn = self.tokenizer.encode(turn, add_special_tokens=False)
                if encoded_turn:
                    input_ids.extend(encoded_turn + [self.tokenizer.eos_token_id])

        # Remove None values from input_ids
        input_ids = [token for token in input_ids if token is not None]

        if input_ids:
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            logger.warning(
                "No valid turns found in the conversation. Returning empty input."
            )
            empty_tensor = torch.tensor([], dtype=torch.long).unsqueeze(0)
            return {"input_ids": empty_tensor, "attention_mask": empty_tensor}

    @staticmethod
    def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        # Filter the logits using top-k and/or top-p filtering
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Convert from sorted indices to original indices
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits
