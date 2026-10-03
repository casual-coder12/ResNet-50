import sys
import os

# Ensure the root directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, utils
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def prepare_mnist_dataset(batch_size=32, buffer_size=10000):
    """
    Loads the MNIST dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    print("MNIST Shapes (Before Preprocessing):", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    X_train = X_train.astype('float32') / 255.0  # Normalize X_train pixel values to [0, 1]
    X_test = X_test.astype('float32') / 255.0  # Normalize X_test pixel values to [0, 1]

    X_train = X_train.reshape(-1, 28, 28, 1)  # Reshape for MNIST (grayscale)
    X_test = X_test.reshape(-1, 28, 28, 1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    X_train = tf.pad(X_train, [[0, 0], [2, 2], [2, 2], [0, 0]])  # Resize to 32x32 for consistency
    X_val = tf.pad(X_val, [[0, 0], [2, 2], [2, 2], [0, 0]])
    X_test = tf.pad(X_test, [[0, 0], [2, 2], [2, 2], [0, 0]])

    print("MNIST Shapes (After Preprocessing):", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

def prepare_cifar10_dataset(batch_size=64, buffer_size=10000):
    """
    Loads the CIFAR-10 dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    dataset = load_dataset("uoft-cs/cifar10")
    
    X_train = np.array(dataset['train']['img'], dtype=np.uint8)
    y_train = np.array(dataset['train']['label'])

    X_test = np.array(dataset['test']['img'], dtype=np.uint8)
    y_test = np.array(dataset['test']['label'])

    print("CIFAR-10 Raw Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Debugging shapes

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

    def normalize_fn(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_dataset = train_dataset.map(
        normalize_fn,
        num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_dataset.map(
        normalize_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = test_dataset.map(
        normalize_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def prepare_imagenette_dataset(batch_size=64, buffer_size=10000):
    """
    Loads the Imagenette dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    dataset = load_dataset("frgfm/imagenette", "160px", trust_remote_code=True)
    
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    image_size = (160, 160)

    X_train = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["train"]["image"]])
    y_train = np.asarray(train_val["train"]["label"], dtype=np.int32)

    X_val = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["test"]["image"]])
    y_val = np.asarray(train_val["test"]["label"], dtype=np.int32)

    X_test = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in dataset["validation"]["image"]])
    y_test = np.asarray(dataset["validation"]["label"], dtype=np.int32)

    print("Imagenette Processed Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=min(buffer_size, len(X_train)))

    def normalize_fn(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_dataset = train_dataset.map(
        normalize_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(
        normalize_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(
        normalize_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

