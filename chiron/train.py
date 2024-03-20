import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import List

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.layers.snn.model import SNNModel


def train(
    model: SNNModel,
    sdr_embeddings: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    conversation_texts: List[List[str]],
    labels: torch.Tensor,
    config: dict,
    device: torch.device,
) -> None:
    """
    Train the SNN model.

    Args:
        model (SNNModel): The SNN model to train.
        sdr_embeddings (torch.Tensor): SDR embeddings tensor.
        adjacency_matrix (torch.Tensor): Adjacency matrix tensor.
        conversation_texts (List[List[str]]): List of conversation texts, where each conversation is a list of strings.
        labels (torch.Tensor): Labels tensor for downstream tasks.
        config (dict): Training configuration.
        device (torch.device): Device to use for training (GPU or CPU).
    """
    # Set device
    model.to(device)

    # Ensure all tensors have the same size along the first dimension
    num_samples = min(sdr_embeddings.size(0), labels.size(0))
    sdr_embeddings = sdr_embeddings[:num_samples]
    labels = labels[:num_samples]

    # Create data loader
    dataset = TensorDataset(sdr_embeddings, labels)
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    num_epochs = config["num_epochs"]
    accumulation_steps = config["accumulation_steps"]
    num_batches = len(dataloader)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )
        for i, (sdr_batch, labels_batch) in enumerate(progress_bar):
            sdr_batch = sdr_batch.to(device)
           #  labels_batch = labels_batch.to(device)
            adjacency_batch = adjacency_matrix.to(device)

            batch_start = i * config["batch_size"]
            batch_end = min((i + 1) * config["batch_size"], len(conversation_texts))
            conversation_texts_batch = conversation_texts[batch_start:batch_end]

            with torch.cuda.amp.autocast():
                outputs = model(sdr_batch, adjacency_batch, conversation_texts_batch)
                targets = sdr_batch[1:]  # Shift the targets by 1 timestep
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * sdr_batch.size(0)

            progress_bar.set_postfix(loss=loss.item())

        train_loss /= num_samples
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        # Evaluate the model
        eval_scores = evaluate(model, dataloader, adjacency_matrix, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Eval Loss: {eval_scores['eval_loss']:.4f}")

        # Perform downstream tasks evaluation
        semantic_sim_accuracy, semantic_sim_f1 = semantic_similarity_prediction(
            sdr_embeddings.cpu().numpy(), labels.cpu().numpy()
        )
        text_class_accuracy, text_class_f1 = text_classification(
            sdr_embeddings.cpu().numpy(), labels.cpu().numpy()
        )
        print(
            f"Semantic Similarity Prediction - Accuracy: {semantic_sim_accuracy:.4f}, F1 Score: {semantic_sim_f1:.4f}"
        )
        print(
            f"Text Classification - Accuracy: {text_class_accuracy:.4f}, F1 Score: {text_class_f1:.4f}"
        )

        # Save the model checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

        # Release unused memory
        torch.cuda.empty_cache()

    print("Training completed.")


def evaluate(
    model: SNNModel,
    dataloader: DataLoader,
    adjacency_matrix: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on the given data loader.

    Args:
        model (SNNModel): The model to evaluate.
        dataloader (DataLoader): The data loader for evaluation data.
        adjacency_matrix (torch.Tensor): Adjacency matrix tensor.
        device (torch.device): The device to run the model on (GPU or CPU).

    Returns:
        dict: Dictionary containing evaluation scores.
    """
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            sdr_batch, _ = batch
            sdr_batch = sdr_batch.to(device)

            adjacency_batch = adjacency_matrix.to(device)

            outputs = model(sdr_batch, adjacency_batch, [])
            targets = sdr_batch[1:]  # Shift the targets by 1 timestep
            loss = nn.MSELoss()(outputs, targets)

            eval_loss += loss.item() * sdr_batch.size(0)

    eval_loss /= len(dataloader.dataset)

    evaluation_scores = {"eval_loss": eval_loss}
    return evaluation_scores