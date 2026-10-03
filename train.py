import os
import sys
import argparse

from models.resnet50 import ResNet50
from data.dataset import prepare_mnist_dataset, prepare_cifar10_dataset, prepare_imagenette_dataset
from utils.trainer import ResNetTrainer
from utils.visualize import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on MNIST or CIFAR-10 datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "imagenette"],
        help="Dataset to use for training (default: mnist).",
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="both",
        choices=["w", "m", "both"],
        help="Type of save operation: 'w' for weights, 'm' for entire model, 'both' for both (default: w).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train (default: 15).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--load_checkpoint", "-lc",
        action="store_true",
        help="Flag to load from a checkpoint if available (default: False).",
    )
    return parser.parse_args()

def main():
    """
    Main function to train the ResNet-50 model on the specified dataset (MNIST or CIFAR-10).
    It prepares the dataset, initializes the model and trainer, and starts the training process.
    """
    args = parse_args()
  
    print(f"=== Starting Training on {args.dataset.upper()} ===")

    # Set input dimensions according to dataset choice
    if args.dataset == "mnist":
        input_shape = (32, 32, 1)
    elif args.dataset == "imagenette":
        input_shape = (160, 160, 3)
    else:
        input_shape = (32, 32, 3)

    # Get the appropriate dataset
    if args.dataset == "mnist":
        train_data, val_data, test_data = prepare_mnist_dataset(batch_size=args.batch_size)
    elif args.dataset == "cifar10":
        train_data, val_data, test_data = prepare_cifar10_dataset(batch_size=args.batch_size)
    elif args.dataset == "imagenette":
        train_data, val_data, test_data = prepare_imagenette_dataset(batch_size=args.batch_size)
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist', 'cifar10', or 'imagenette'.")

    steps_per_epoch = len(train_data)
    total_steps = args.epochs * steps_per_epoch

    # Instantiate model and trainer wrapper
    model = ResNet50(input_shape=input_shape, num_classes=10)
    trainer = ResNetTrainer(model=model, learning_rate=args.learning_rate, optimizer='sgd', steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    # Train model
    history = trainer.train(train_data=train_data, val_data=val_data, epochs=args.epochs, dataset_name=args.dataset, save_type=args.save_type, load_checkpoint=args.load_checkpoint)

    plot_training_history(history, dataset_name=args.dataset)

    # Quick evaluation on test data
    results = trainer.evaluate(test_data=test_data, dataset_name=args.dataset)
    print("\n" + "="*50)
    print(f"--- Results on Test Data (for detailed evaluation - run evaluate.py) ---")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()