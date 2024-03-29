import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from chiron.evaluation.metrics import evaluate_text_prediction
from chiron.layers.snn.model import SNNModel


def train(
    model: SNNModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: PreTrainedTokenizer,
    config: dict,
    device: torch.device,
    adjacency_matrix_sparse: torch.sparse_coo_tensor,
    writer: SummaryWriter,
) -> None:
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
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss()
    num_classes = (
        model.output_size
    )  # Assuming output_size represents the number of classes
    node_to_class_mapping = {i: i % num_classes for i in range(num_classes)}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["patience"], factor=0.1, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # Set the number of accumulation steps
    accumulation_steps = config["accumulation_steps"]

    # Enable mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
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
            if batch_idx % accumulation_steps == 0:
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

        # Evaluate on the validation set
        val_loss = evaluate(
            model, val_dataloader, tokenizer, device, adjacency_matrix_sparse
        )

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
                logger.info(f"Early stopping after {epoch + 1} epochs.")
                break

        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )


def evaluate(
    model: SNNModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    adjacency_matrix_sparse: torch.sparse_coo_tensor,
) -> float:
    model.eval()
    criterion = nn.NLLLoss()
    num_classes = (
        model.output_size
    )  # Assuming output_size represents the number of classes
    node_to_class_mapping = {i: i % num_classes for i in range(num_classes)}
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get the input tensors and labels
            input_ids, attention_mask, label, node_indices = batch

            # Move the input tensors and labels to the device
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

            # Forward pass
            outputs = model(
                input_ids, attention_mask, adjacency_matrix_sparse, node_indices
            )

            # Reshape outputs to match the shape of class_indices
            outputs = outputs.view(-1, num_classes)

            # Compute the loss
            loss = criterion(outputs, class_indices)

            # Accumulate the loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    text_prediction_metrics = evaluate_text_prediction(
        model, tokenizer, dataloader.dataset, device
    )
    logger.info(f"Text Prediction Metrics: {text_prediction_metrics}")

    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}")

    return avg_loss
