
import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chiron.evaluation.downstream_tasks import (
    semantic_similarity_prediction,
    text_classification,
)
from chiron.evaluation.visualization import plot_evaluation_metrics
from chiron.layers.sdr.sdr_generation import SDRGenerator
from chiron.layers.snn.model import SNNModel
from chiron.preprocessing.embedding import Word2VecEmbedding
from chiron.preprocessing.text_preprocessing import TextPreprocessor
from chiron.train import train, evaluate
from chiron.utils.config import Config


def main(config_path: str) -> None:
    """
    Main function to run the Semantic Folding training pipeline.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = Config(config_path)

    # Load dataset
    from datasets import load_dataset

    dataset = load_dataset(
        config["dataset_params"]["dataset_name"],
        config["dataset_params"]["dataset_config"],
        split=config["dataset_params"]["split"],
    )
    conversations = dataset["conversations"]

    # Preprocess text
    preprocessor = TextPreprocessor(**config["preprocessing_params"])
    preprocessed_conversations = []
    for conv in conversations:
        preprocessed_conv = [preprocessor.preprocess(turn["value"]) for turn in conv]
        preprocessed_conversations.append(preprocessed_conv)

    # Generate word embeddings
    embedding_model = Word2VecEmbedding(**config["embedding_params"])
    embeddings = []
    for conv in preprocessed_conversations:
        conv_embeddings = [embedding_model.generate_embeddings(turn) for turn in conv]
        embeddings.append(conv_embeddings)

    # Generate SDRs
    sdr_generator = SDRGenerator(**config["sdr_params"])
    sdr_embeddings = []
    for conv_embeddings in embeddings:
        conv_sdr_embeddings = []
        for turn_embeddings in conv_embeddings:
            turn_sdr_embeddings = sdr_generator.generate_sdr_embeddings(turn_embeddings)
            conv_sdr_embeddings.append(turn_sdr_embeddings)
        sdr_embeddings.append(conv_sdr_embeddings)

    # Convert SDR embeddings to a list of tensors
    sdr_embeddings_list = []
    for i, conv_sdr_embeddings in enumerate(sdr_embeddings):
        for j, turn_sdr_embeddings in enumerate(conv_sdr_embeddings):
            if len(sdr_embeddings_list) < len(conv_sdr_embeddings):
                sdr_embeddings_list.append([])
            sdr_embeddings_list[j].append(torch.FloatTensor(turn_sdr_embeddings))

    # Create an adjacency matrix
    adjacency_matrix_tensor = torch.ones((len(sdr_embeddings_list), len(sdr_embeddings_list)))

    # Create SNN model
    snn_model = SNNModel(
        sp_params=config["htm_params"],
        snn_params=config["snn_params"],
        gat_params=config["gat_params"],
        encoder_params=config["encoder_params"],
        decoder_params=config["decoder_params"],
    )

    # Train model
    device = config["device"]
    train_config = {
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "learning_rate": config["learning_rate"],
    }
    train(snn_model, sdr_embeddings_list, train_config, device, adjacency_matrix_tensor)

    # Create data loader for evaluation
    dataset = TensorDataset(*sdr_embeddings_list, adjacency_matrix_tensor)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])

    # Evaluate model and plot metrics
    evaluation_scores: Dict[str, float] = evaluate(snn_model, dataloader, device)
    semantic_sim_accuracy, semantic_sim_f1 = semantic_similarity_prediction(
        sdr_embeddings_list, config["labels"]
    )
    text_class_accuracy, text_class_f1 = text_classification(
        sdr_embeddings_list, config["labels"]
    )
    plot_evaluation_metrics(
        evaluation_scores,
        semantic_sim_accuracy,
        semantic_sim_f1,
        text_class_accuracy,
        text_class_f1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Folding Training")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )
    args = parser.parse_args()

    main(args.config)