import os
import datetime
import safetensors
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from chiron.layers.snn.model import SNNModel

from chiron.evaluation.metrics import evaluate_text_prediction


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
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Update the latest checkpoint symlink
    latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
    os.makedirs(latest_checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(
        latest_checkpoint_dir,
        f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )
    os.symlink(os.path.abspath(checkpoint_path), latest_checkpoint_path)
    logger.info(f"Latest checkpoint symlink updated: {latest_checkpoint_path}")


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
    adjacency_matrix_sparse: torch.sparse_coo_tensor,
    writer: SummaryWriter,
    checkpoint_dir: str = ".checkpoints",
    resume_from_latest: bool = True,
    model_name: str = "snn_model",
) -> tuple:
    """
    Train the SNNModel on the given datasets.

    Args:
        model (SNNModel): The model to train.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        val_dataloader (torch.utils.data.DataLoader): The validation data loader.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        config (dict): The training configuration.
        device (torch.device): The device to run the training on.
        adjacency_matrix_sparse (torch.sparse_coo_tensor): The adjacency matrix in sparse COO format.
        writer (SummaryWriter): The TensorBoard writer for logging.
        checkpoint_dir (str): The directory to save checkpoints.
        resume_from_latest (bool): Whether to resume training from the latest checkpoint.
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing lists of training and validation losses for each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss()
    num_classes = model.output_size
    node_to_class_mapping = {i: i % num_classes for i in range(num_classes)}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["patience"], factor=0.1, verbose=True
    )

    start_epoch = 1
    if resume_from_latest:
        latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
        latest_checkpoint_files = os.listdir(latest_checkpoint_dir)
        if latest_checkpoint_files:
            latest_checkpoint_path = sorted(latest_checkpoint_files)[-1]
            latest_checkpoint_path = os.path.join(
                latest_checkpoint_dir, latest_checkpoint_path
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

    # Enable mixed-precision training
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
            input_ids, attention_mask, label, node_indices = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            node_indices = node_indices.to(device)

            # Clamp node_indices to the valid range
            node_indices = node_indices.clamp(min=0, max=num_classes - 1)

            # Convert node_indices to class indices using the mapping
            class_indices = torch.tensor(
                [node_to_class_mapping[idx.item()] for idx in node_indices],
                device=device,
            )

            # Mixed-precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(
                    input_ids, attention_mask, adjacency_matrix_sparse, node_indices
                )

                # Reshape outputs to match the shape of class_indices
                outputs = outputs.view(-1, num_classes)

                # Compute the loss
                loss = criterion(outputs, class_indices)
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
        val_loss = evaluate(
            model, val_dataloader, tokenizer, device, adjacency_matrix_sparse
        )
        val_losses.append(val_loss)

        # Save the model as a SafeTensor
        safetensors.torch.save_safetensors(
            f"{model_name}_epoch_{epoch}.safetensor", model.state_dict()
        )

        # Save the checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir, model_name)

        # Log the losses
        writer.add_scalar("Loss/Train", train_loss, epoch - 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch - 1)

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
    safetensors.torch.save_safetensors(
        f"{model_name}_final.safetensor", model.state_dict()
    )

    # Save the final checkpoint
    save_checkpoint(
        config["num_epochs"], model, optimizer, scheduler, checkpoint_dir, model_name
    )

    return train_losses, val_losses


def evaluate(
    model: SNNModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    adjacency_matrix_sparse: torch.sparse.Tensor,
) -> float:
    """
    Evaluate the model on the given dataset.
    Args:
        model (SNNModel): The model to evaluate.
        dataloader (DataLoader): The data loader for evaluation.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        device (torch.device): The device to run the evaluation on.
        adjacency_matrix_sparse (torch.sparse.Tensor): The adjacency matrix in sparse format.
    Returns:
        float: The average loss on the evaluation dataset.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")  # Use mean reduction
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get the input tensors and labels
            input_ids, attention_mask, label, node_indices = batch

            # Move the input tensors and labels to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            node_indices = node_indices.to(device)

            # Ensure the label tensor is 1D
            label = label.squeeze(1)

            # Forward pass
            outputs = model(
                input_ids, attention_mask, adjacency_matrix_sparse, node_indices
            )

            # Compute the loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), label.view(-1))

            # Accumulate the loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    text_prediction_metrics = evaluate_text_prediction(
        model, tokenizer, dataloader.dataset, device, adjacency_matrix_sparse
    )
    logger.info(f"Text Prediction Metrics: {text_prediction_metrics}")

    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}")

    return avg_loss
