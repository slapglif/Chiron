from typing import Dict, Any, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.layers.snn.model import SNNModel, create_adjacency_matrix


def train(
        model: SNNModel,
        sdr_embeddings: List[List[float]],
        config: Dict[str, Any],
        device: torch.device,
) -> None:
    # Set device
    model.to(device)

    # Create data loader
    sdr_embeddings_tensor = torch.FloatTensor(np.array(sdr_embeddings))
    adjacency_matrix_tensor = torch.FloatTensor(create_adjacency_matrix(sdr_embeddings))
    print(f"SDR Embeddings Shape: {sdr_embeddings_tensor.shape}")
    print(f"Adjacency Matrix Shape: {adjacency_matrix_tensor.shape}")

    dataset = TensorDataset(sdr_embeddings_tensor, adjacency_matrix_tensor)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

    # Define loss function, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = len(dataloader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    # Gradient accumulation steps
    accumulation_steps = 4

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}", unit="batch")):
            sdr_batch, adjacency_batch = batch

            with torch.cuda.amp.autocast():
                outputs = model(sdr_batch, adjacency_batch)
                targets = sdr_batch[1:]  # Shift the targets by 1 timestep
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * sdr_batch.size(0)

                train_loss /= len(dataset)
                print(f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}")  # noqa: E501

                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    eval_scores = evaluate(model, dataloader, device)
                    print(f"Epoch {epoch + 1}/{config['num_epochs']} - Eval Loss: {eval_scores['eval_loss']:.4f}")

                # Save the model checkpoint
                torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pt")  # noqa: E501

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item() * sdr_batch.size(0)

                train_loss /= len(dataset)
                print(f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}")  # noqa: E501

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            eval_scores = evaluate(model, dataloader, device)
            print(f"Epoch {epoch + 1}/{config['num_epochs']} - Eval Loss: {eval_scores['eval_loss']:.4f}")

            # Perform downstream tasks evaluation
            sdr_embeddings_tensor = sdr_embeddings_tensor.to(device)
            adjacency_matrix_tensor = adjacency_matrix_tensor.to(device)
            outputs = model(sdr_embeddings_tensor, adjacency_matrix_tensor)
            semantic_sim_accuracy, semantic_sim_f1 = semantic_similarity_prediction(
                outputs.cpu().numpy(), config["labels"]
            )
            text_class_accuracy, text_class_f1 = text_classification(
                outputs.cpu().numpy(), config["labels"]
            )
            print(
                f"Semantic Similarity Prediction - Accuracy: {semantic_sim_accuracy:.4f}, F1 Score: {semantic_sim_f1:.4f}"  # noqa: E501
                # noqa: E501
            )
            print(
                f"Text Classification - Accuracy: {text_class_accuracy:.4f}, F1 Score: {text_class_f1:.4f}"  # noqa: E501
                # noqa: E501
            )
            # Save the model checkpoint
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pt")

            # Release unused cached memory
            torch.cuda.empty_cache()


def evaluate(
        model: SNNModel, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model on the given data loader.
    Args:
    model (SNNModel): The model to evaluate.
    dataloader (DataLoader): The data loader for evaluation data.
    device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary containing evaluation scores.
    """
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            sdr_batch, adjacency_batch = batch
            sdr_batch, adjacency_batch = sdr_batch.to(device), adjacency_batch.to(device)  # noqa: E501

            outputs = model(sdr_batch, adjacency_batch)
            targets = sdr_batch[1:]  # Shift the targets by 1 timestep
            loss = torch.nn.MSELoss()(outputs, targets)

            eval_loss += loss.item() * sdr_batch.size(0)

    eval_loss /= len(dataloader.dataset)

    evaluation_scores = {"eval_loss": eval_loss}
    return evaluation_scores
