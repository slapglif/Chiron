# chiron/train.py

import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Union

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
            attention_mask, padded labels, and node_indices.
    """
    input_ids, attention_mask, labels, node_indices = zip(*batch)

    # Pad input_ids and attention_mask
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    # Pad labels to ensure they have the same sequence length as input_ids
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Convert node_indices to a tensor with the correct shape
    batch_size = len(batch)
    node_indices_tensor = torch.tensor(node_indices, dtype=torch.long).view(
        batch_size, -1
    )

    return padded_input_ids, padded_attention_mask, padded_labels, node_indices_tensor


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
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of model parameters.

    After each optimizer step, call update() to refresh the shadow parameters.
    Use apply() before evaluation to swap in EMA weights, and restore() afterward
    to return to the training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        # Initialize shadow parameters as a copy of the current model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters using exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """Replace model parameters with EMA shadow parameters for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with linear warmup for the first num_warmup_steps, then
    cosine decay to 0 for the remaining steps.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler with the combined warmup + cosine decay schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: SNNModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    config: dict,
    device: torch.device,
    adjacency_matrix: Union[scipy.sparse.csr_matrix, torch.Tensor],
    writer: SummaryWriter,
    checkpoint_dir: str = ".checkpoints",
    resume_from_latest: bool = True,
    model_name: str = "snn_model",
) -> Tuple[List[float], List[float]]:
    """
    Train the SNNModel on the given dataset.

    Uses AdamW with decoupled weight decay, cosine annealing with linear warmup,
    mixed-precision training via modern torch.amp API, exponential moving average
    of model weights, gradient accumulation, and gradient clipping.

    Args:
        model (SNNModel): The model to train.
        train_dataloader (DataLoader): The data loader for the training set.
        val_dataloader (DataLoader): The data loader for the validation set.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        config (dict): The configuration dictionary containing training parameters.
        device (torch.device): The device to use for training.
        adjacency_matrix (Union[scipy.sparse.csr_matrix, torch.Tensor]): The adjacency matrix as a SciPy sparse matrix or a PyTorch tensor.
        writer (SummaryWriter): The TensorBoard writer for logging.
        checkpoint_dir (str, optional): The directory to save checkpoints. Defaults to ".checkpoints".
        resume_from_latest (bool, optional): Whether to resume training from the latest checkpoint. Defaults to True.
        model_name (str, optional): The name of the model for checkpointing. Defaults to "snn_model".

    Returns:
        Tuple[List[float], List[float]]: A tuple containing lists of training and validation losses.
    """
    # AdamW optimizer with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    # Consistent loss function: CrossEntropyLoss with label smoothing for both
    # training and evaluation. ignore_index=-100 skips padding tokens.
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    # Compute total training steps and warmup steps for the cosine schedule
    steps_per_epoch = len(train_dataloader)
    num_epochs = config["num_epochs"]
    total_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = int(0.1 * total_training_steps)  # 10% warmup

    # Cosine annealing with linear warmup scheduler
    scheduler = _get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Gradient accumulation steps
    accumulation_steps = config["accumulation_steps"]

    # Initialize Exponential Moving Average of model weights
    ema = ExponentialMovingAverage(model, decay=0.999)

    # Initialize GradScaler for mixed-precision training (modern API)
    scaler = torch.amp.GradScaler("cuda")

    # Resume from checkpoint if requested
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
            # Re-initialize EMA shadow from the loaded model weights
            ema = ExponentialMovingAverage(model, decay=0.999)
            logger.info(
                f"Resuming training from the latest checkpoint: {latest_checkpoint_path}"
            )
        else:
            logger.info(
                "No previous checkpoints found. Starting training from scratch."
            )

    best_val_loss = float("inf")
    patience_counter = 0

    # Lists to store per-epoch losses
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        global_grad_norm = 0.0

        # Zero the gradients at the start of each epoch
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(
            tqdm(
                train_dataloader,
                desc="Train",
                total=len(train_dataloader),
                leave=False,
            ),
            start=1,
        ):
            # Unpack batch and move to device
            input_ids, attention_mask, labels, node_indices = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)

            # Convert adjacency matrix to dense tensor if necessary
            if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
                adj_matrix_tensor = torch.tensor(
                    adjacency_matrix.toarray(), dtype=torch.float32, device=device
                )
            else:
                adj_matrix_tensor = adjacency_matrix.to(device)

            # Mixed-precision forward pass (modern API)
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids, attention_mask, adj_matrix_tensor, node_indices
                )

                # Mask out padding tokens and compute loss
                non_pad_mask = labels.view(-1) != -100
                loss = criterion(
                    outputs.view(-1, outputs.size(-1))[non_pad_mask],
                    labels.view(-1)[non_pad_mask],
                )
                # Normalize loss by accumulation steps so the effective gradient
                # magnitude is independent of the accumulation count
                loss = loss / accumulation_steps

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Step optimizer every accumulation_steps batches
            if batch_idx % accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                global_grad_norm = grad_norm.item()

                # Optimizer step and scaler update
                scaler.step(optimizer)
                scaler.update()

                # Step the warmup + cosine scheduler per optimizer step
                scheduler.step()

                # Update EMA shadow weights
                ema.update()

                # Zero gradients for next accumulation window
                optimizer.zero_grad()

            # Accumulate raw loss (undo the division for reporting)
            epoch_loss += loss.item() * accumulation_steps

        # Handle any remaining gradients if the last batch didn't align with
        # the accumulation boundary
        if batch_idx % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            global_grad_norm = grad_norm.item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()
            optimizer.zero_grad()

        # Compute average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Save the model as a SafeTensor
        safetensors.torch.save_file(
            model.state_dict(),
            f"{model_name}_epoch_{epoch}_pre_eval.safetensors",
        )

        # Save the checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir, model_name)

        # Evaluate on the validation set using EMA weights
        ema.apply()
        val_loss = evaluate(model, val_dataloader, tokenizer, device, adjacency_matrix)
        ema.restore()
        val_losses.append(val_loss)

        # Retrieve current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("GradNorm", global_grad_norm, epoch)

        logger.info(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.2e}, "
            f"Grad Norm: {global_grad_norm:.4f}, "
            f"EMA: active (decay=0.999)"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"Early stopping after {epoch} epochs.")
                break

    # Visualize the network architecture after training
    model.visualize()

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

    Uses the same CrossEntropyLoss (with label smoothing) as training for
    consistent loss computation.

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
    # Same loss function as training for consistency
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch and move to device
            input_ids, attention_mask, labels, node_indices = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)

            # Convert adjacency matrix to dense tensor if necessary
            if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
                adj_matrix_tensor = torch.tensor(
                    adjacency_matrix.toarray(), dtype=torch.float32, device=device
                )
            else:
                adj_matrix_tensor = adjacency_matrix.to(device)

            # Forward pass
            outputs = model(
                input_ids, attention_mask, adj_matrix_tensor, node_indices
            )

            # Compute loss with the same masking approach as training
            non_pad_mask = labels.view(-1) != -100
            loss = criterion(
                outputs.view(-1, outputs.size(-1))[non_pad_mask],
                labels.view(-1)[non_pad_mask],
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    text_prediction_metrics = evaluate_text_prediction(
        model, tokenizer, dataloader.dataset, device, adjacency_matrix
    )
    logger.info(f"Text Prediction Metrics: {text_prediction_metrics}")

    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}")

    return avg_loss
