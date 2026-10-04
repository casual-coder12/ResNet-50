# ResNet-50 Image Classification

A TensorFlow/Keras implementation of ResNet-50 for image classification, with training and evaluation pipelines for MNIST and CIFAR-10.

## Overview

This project implements a 50-layer residual network using bottleneck blocks and skip connections. The training pipeline supports grayscale and RGB inputs, data augmentation, model and weight saving, evaluation, and training visualizations.

## Features

- **ResNet-50 architecture**: Bottleneck residual blocks with projection shortcuts where dimensions change.
- **MNIST and CIFAR-10 support**: MNIST uses one input channel; CIFAR-10 uses three.
- **Resolution-dependent stem**: The model selects a different initial convolution and downsampling path based on the configured input height.
- **Data augmentation**: Random crop, horizontal flip, rotation, and translation are included in the model.
- **Optimizer options**: The trainer supports SGD with momentum and cosine learning-rate decay, or Adam. The current `train.py` configuration selects SGD; the two optimizers are alternatives, not used simultaneously.
- **Training and evaluation utilities**: Validation, accuracy and top-5 accuracy metrics, best-validation-loss checkpoints, CSV history, and plots.
- **Flexible saving**: Save model weights, the complete Keras model, or both.

## Architecture

The network follows the ResNet-50 bottleneck layout:

| Stage | Bottleneck blocks | Filters per block |
|---|---:|---|
| 2 | 3 | 64, 64, 256 |
| 3 | 4 | 128, 128, 512 |
| 4 | 6 | 256, 256, 1024 |
| 5 | 3 | 512, 512, 2048 |

The stem is selected using the image height in `input_shape`:

| Configured input height | Initial layers |
|---|---|
| Below 64 | 3×3 convolution, stride 1; no initial max pooling |
| 64 to 127 | 3×3 convolution, stride 1; 2×2 max pooling |
| 128 or above | 7×7 convolution, stride 2; 3×3 max pooling |

After the residual stages, the current model applies max pooling, dropout, flattening, and a 10-class softmax output layer.

### Input-size note

The model currently applies `RandomCrop(32, 32)` before the resolution-dependent stem. Consequently, inputs larger than 32×32 are cropped to 32×32 before the convolutional layers, even though the stem is selected from the original `input_shape`. To preserve larger image resolutions, adjust the crop size or make cropping conditional on the input size.

## Data Augmentation

The model includes the following Keras preprocessing layers:

- Random crop to 32×32
- Random horizontal flip
- Random rotation with factor `0.05`
- Random translation with factors `0.1` vertically and horizontally

Keras preprocessing augmentation layers are active during training and are bypassed during inference. The same augmentation sequence is currently used for both MNIST and CIFAR-10.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

Train on MNIST (the default dataset):

```bash
python train.py --dataset mnist
```

Train on CIFAR-10:

```bash
python train.py --dataset cifar10 --epochs 20
```

Training options include `--epochs`, `--batch_size`, `--learning_rate`, and `--save_type` (`w`, `m`, or `both`). Run `python train.py --help` to see the current defaults and all available options.

### Evaluate

Evaluate saved MNIST weights:

```bash
python evaluate.py --dataset mnist --load_type w
```

Evaluate a saved complete CIFAR-10 model:

```bash
python evaluate.py --dataset cifar10 --load_type m
```

Evaluate a saved best CIFAR-10 model:

```bash
python evaluate.py --dataset cifar10 --load_type b
```

## Optimizers

`ResNetTrainer` supports two optimizer configurations:

- **SGD**: momentum `0.9` and a cosine-decay learning-rate schedule. This is the optimizer selected by the current training script.
- **Adam**: uses the configured learning rate and is available as the trainer's default when SGD is not selected.

Each training run uses one optimizer. The current command-line script does not expose an optimizer flag; change the `optimizer_name` argument passed to `ResNetTrainer` to select Adam instead of SGD.

## Project Structure

```text
.
├── data/
│   └── dataset.py             # MNIST and CIFAR-10 loading and preprocessing
├── models/
│   ├── resnet50.py            # ResNet-50 architecture and augmentation
├── utils/
│   ├── trainer.py             # Training, saving, loading, and evaluation
│   └── visualize.py           # Training and evaluation visualizations
├── saved_models/              # Saved models, weights, and training history
├── outputs/                   # Generated plots and evaluation outputs
├── train.py                   # Training entry point
├── evaluate.py                # Evaluation entry point
└── requirements.txt           # Python dependencies
```

## Results

Results for the current run can be viewed in the `ResNet50.ipynb` notebook. Imagenette results will be added once training is executed.

| Dataset | Accuracy | Loss |
|---|---|---|
| CIFAR-10 | ~76% | ~0.73 |
| Imagenette | ~65.5% | ~1.08 |

## Generated Files

Training artifacts are stored in `saved_models/`, including model weights or complete models, a best-validation-loss checkpoint, and CSV training history. Plots and evaluation visualizations are written to `outputs/`.

## Dependencies

The main dependencies are TensorFlow/Keras, NumPy, Matplotlib, Seaborn, and scikit-learn. See `requirements.txt` for the complete list.

---

*Neural Networks and Deep Learning — ResNet-50 Project*
