import os
from typing import List, Tuple, Union

import safetensors
import scipy
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from chiron.evaluation.metrics import evaluate_text_prediction
from chiron.layers.snn.model import SNNModel


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences in a batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]): List of samples from the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the padded input_ids,
            attention_mask, stacked labels, and node_indices.
    """
    input_ids, attention_mask, labels, node_indices = zip(*batch)

    # Pad input_ids and attention_mask
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    # Stack labels instead of padding
    stacked_labels = torch.stack(labels)

    # Convert node_indices to tensor
    node_indices = torch.tensor(node_indices, dtype=torch.long)

    return padded_input_ids, padded_attention_mask, stacked_labels, node_indices


def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir, model_name):
    """
    Save the model state, optimizer state, and scheduler state as a checkpoint.

    Args:
        epoch (int): The current epoch number.
        model (SNNModel): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler instance.
        checkpoint_dir (str): The directory to save the checkpoint.
        model_name (str): The name of the model.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load a checkpoint and update the model, optimizer, and scheduler states.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        model (SNNModel): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler instance.

    Returns:
        int: The epoch to start training from.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


def train(
    model: SNNModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: PreTrainedTokenizer,
    config: dict,
    device: torch.device,
    adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor],
    writer: SummaryWriter,
    checkpoint_dir: str = ".checkpoints",
    resume_from_latest: bool = True,
    model_name: str = "snn_model",
) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["patience"], factor=0.1, verbose=True
    )

    start_epoch = 1
    if resume_from_latest:
        latest_checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
            reverse=True,
        )
        if latest_checkpoint_files:
            latest_checkpoint_path = os.path.join(
                checkpoint_dir, latest_checkpoint_files[0]
            )
            start_epoch = load_checkpoint(
                latest_checkpoint_path, model, optimizer, scheduler
            )
            logger.info(
                f"Resuming training from the latest checkpoint: {latest_checkpoint_path}"
            )
        else:
            logger.info(
                "No previous checkpoints found. Starting training from scratch."
            )

    best_val_loss = float("inf")
    patience_counter = 0

    # Set the number of accumulation steps
    accumulation_steps = config["accumulation_steps"]

    # Initialize the GradScaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        logger.info(f"Epoch {epoch}/{config['num_epochs']}")
        train_loss = 0.0
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc="Train"), start=1
        ):
            # Get the input tensors and labels
            input_ids, attention_mask, labels, node_indices = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)

            # Convert the adjacency matrix to a dense PyTorch tensor if necessary
            if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
                adj_matrix_tensor = torch.tensor(
                    adjacency_matrix.toarray(), dtype=torch.float32, device=device
                )
            else:
                adj_matrix_tensor = adjacency_matrix.to(device)

            # Mixed-precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(
                    input_ids, attention_mask, adj_matrix_tensor, node_indices
                )

                # Reshape the labels tensor to match the expected shape
                labels = labels.view(-1, outputs.size(-1))

                # Compute the loss
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            # Backward pass with mixed-precision
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Accumulate the loss
            train_loss += loss.item()

        # Compute the average training loss
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Evaluate on the validation set
        val_loss = evaluate(model, val_dataloader, tokenizer, device, adjacency_matrix)
        val_losses.append(val_loss)

        # Save the model as a SafeTensor
        safetensors.torch.save_file(
            model.state_dict(), f"{model_name}_epoch_{epoch}.safetensors"
        )

        # Save the checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir, model_name)

        # Log the losses
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Update the learning rate scheduler
        scheduler.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"Early stopping after {epoch} epochs.")
                break

        logger.info(
            f"Epoch {epoch}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Save the final model as a SafeTensor
    safetensors.torch.save_file(model.state_dict(), f"{model_name}_final.safetensors")

    return train_losses, val_losses


def evaluate(
    model: SNNModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor],
) -> float:
    """
    Evaluate the model on the given dataset.

    Args:
        model (SNNModel): The model to evaluate.
        dataloader (DataLoader): The data loader for evaluation.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        device (torch.device): The device to run the evaluation on.
        adjacency_matrix (Union[scipy.sparse.csr_matrix, torch.Tensor]): The adjacency matrix as a SciPy sparse matrix or a PyTorch tensor.

    Returns:
        float: The average loss on the evaluation dataset.
    """
    model.eval()
    criterion = nn.NLLLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get the input tensors and labels
            input_ids, attention_mask, labels, node_indices = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)

            # Convert the adjacency matrix to a dense PyTorch tensor if necessary
            if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
                adj_matrix_tensor = torch.tensor(
                    adjacency_matrix.toarray(), dtype=torch.float32, device=device
                )
            else:
                adj_matrix_tensor = adjacency_matrix.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, adj_matrix_tensor, node_indices)
            model.visualize_network()
            model.visualize_data_flow(input_ids)
            # Reshape the labels tensor to match the expected shape
            labels = labels.view(-1, outputs.size(-1))

            # Compute the loss
            loss = criterion(outputs, labels)

            # Accumulate the loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    text_prediction_metrics = evaluate_text_prediction(
        model, tokenizer, dataloader.dataset, device, adjacency_matrix
    )
    logger.info(f"Text Prediction Metrics: {text_prediction_metrics}")

    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}")

    return avg_loss
