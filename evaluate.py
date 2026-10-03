import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses, metrics, models

from data.dataset import prepare_mnist_dataset, prepare_cifar10_dataset
from models.resnet50 import ResNet50
from models.model_loader import load_model
from utils.trainer import ResNetTrainer
from utils.visualize import plot_confusion_matrix, plot_sample_predictions, plot_training_hist_df

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained ResNet-50 model on MNIST or CIFAR-10 datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset name to evaluate on"
    )
    parser.add_argument(
        "--load_type",
        type=str,
        default="w",
        choices=["w", "m", "b"],
        help="Type of load operation: 'w' for weights, 'm' for entire model or 'b' for best model (default: w).",
        )
    return parser.parse_args()

def evaluate():
    """
    Evaluate the model on the test dataset and print the results.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_data (tf.data.Dataset): The test dataset.
    """
    args = parse_args()
    dataset_name = args.dataset

    outputs_dir = "outputs"
    saved_models_dir = "saved_models"
    os.makedirs(outputs_dir, exist_ok=True)
    
    if dataset_name == "mnist":
        _, _, test_data = prepare_mnist_dataset(batch_size=64)
        input_shape = (32, 32, 1)
    elif dataset_name == "cifar10":
        _, _, test_data = prepare_cifar10_dataset(batch_size=64)
        input_shape = (32, 32, 3)
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'cifar10'.")

    # Load the trained model
    model = load_model(
        model_class=ResNet50,
        dataset_name=dataset_name,
        load_type=args.load_type,
        input_shape=input_shape,
        num_classes=10,
        saved_models_dir=saved_models_dir
    )

    trainer = ResNetTrainer(model=model)

    # Evaluate the model on the test dataset
    results = trainer.evaluate(test_data, dataset_name)

    print("\n" + "="*50)
    print(" " * 15 + "Evaluation Results" + " " * 15)
    print("=" * 50 + "\n")
    print(f"Test Loss: {results}")
    print(f"Test Loss: {results["loss"]:.4f}")
    print(f"Test Accuracy: {results["accuracy"]*100:.2f}%")

    history_path = os.path.join(saved_models_dir, f"resnet50_{dataset_name}_history.csv")

    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        plot_training_hist_df(df_history, dataset_name=dataset_name)

    plot_confusion_matrix(model, test_data, dataset_name=dataset_name)

    plot_sample_predictions(model, test_data, dataset_name=dataset_name)

if __name__ == "__main__":
    evaluate()