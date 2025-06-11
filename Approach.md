<h1 align="center">Approach to this project</h1>

## Table of Contents

- [Necessary Imports](#necessary-imports)
- [get_dataset Function](#get_dataset-function)
- [get_model Function](#get_model-function)
- [Training and Evaluation Functions](#training-and-evaluation-functions)
- [plot_metrics Function](#plot_metrics-function)
- [run_experiment Function](#run_experiment-function)
- [Running Each Model on Different Datasets](#running-each-model-on-different-datasets)

---

## Necessary Imports

First Install dependencies:

```bash
!pip install timm -q
```

Then import the necessary libraries:

```python
import os, time, copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
```

## get_dataset Function

`get_dataset` loads and prepares common image datasets (MNIST, FashionMNIST, CIFAR10) for PyTorch. It:

- Accepts `name`, `batch_size`, and `data_dir`, `train_fraction`.
- Selects the dataset, sets `num_classes`, and applies appropriate transforms (resize, tensor conversion, normalization).
- Loads train and validation/test splits, downloading if needed.
- Returns PyTorch DataLoaders for train/val and the number of classes.

---

## get_model Function

`get_model` creates a neural network model based on the given name:

- If `lenet5`, returns a custom LeNet5 implementation (supports grayscale or RGB).
- Otherwise, uses the `timm` and `torchvision` library to load popular models (e.g., AlexNet, ResNet, GoogleNet), with options for pretrained weights and custom output classes.
- Raises an error for unsupported names.

---

## Training and Evaluation Functions

### train_one_epoch

Trains the model for one epoch:

- Loops over batches, moves data to device, computes loss, backpropagates, and updates weights.
- Returns average training loss.

### evaluate

Evaluates the model:

- Disables gradients, loops over validation batches, computes loss and predictions.
- Calculates average loss, accuracy, precision, recall, F1-score, and returns all true/predicted labels.

---

## plot_metrics Function

`plot_metrics` visualizes training history:

- Plots training/validation loss and validation accuracy over epochs using matplotlib.

---

## run_experiment Function

`run_experiment` manages the full training workflow:

- Loads data and model, adapts LeNet5 input channels if needed.
- Sets up loss and optimizer.
- Trains for specified epochs, tracks metrics, saves the best model.
- Plots metrics and prints a classification report.
- Returns the trained model.

## Running Each Model on Different Datasets

To compare models, run each one on each dataset and record their results. This highlights how architectures perform across tasks.

