# Approach.md

## Table of Contents

[Explanation of the `get_dataset` Function](#explanation-of-the-get_dataset-function)
[Explanation of the `get_model` Function](#explanation-of-the-get_model-function)
[Training and Evaluation Functions](#training-and-evaluation-functions)
---

## Explanation of the `get_dataset` Function

This Python function, `get_dataset`, is designed to simplify the process of loading and preparing common image datasets for use in machine learning tasks, particularly with libraries like PyTorch.

Here's a breakdown of what it does:

1. **Function Definition:** It defines a function `get_dataset` that takes three arguments:
    * `name`: A string specifying the name of the dataset to load (e.g., 'mnist', 'fmnist', 'cifar10').
    * `batch_size`: An integer specifying the number of samples in each batch when loading data.
    * `data_dir`: A string specifying the directory where the dataset files should be stored or loaded from. It defaults to './data'.

2. **Dataset Selection and Configuration:**
    * It converts the input `name` to lowercase for case-insensitive matching.
    * It uses `if/elif/else` statements to check the value of `name`.
    * Based on the dataset name, it sets:
        * `num_classes`: The number of output classes for the dataset (e.g., 10 for MNIST, FashionMNIST, and CIFAR10).
        * `transform`: A sequence of image transformations to apply to the dataset. These transformations typically include resizing the images to a standard size (32x32 in this case, except for CIFAR10 which is already 32x32), converting them to PyTorch tensors, and normalizing the pixel values using pre-calculated mean and standard deviation for each dataset. Normalization helps in stabilizing training by ensuring pixel values are within a consistent range, which can improve convergence and performance.
        * `cls`: The specific dataset class from `torchvision.datasets` corresponding to the requested name (e.g., `torchvision.datasets.MNIST`).

3. **Error Handling:** If the provided `name` does not match any of the supported datasets ('mnist', 'fmnist', 'fashionmnist', 'cifar10'), it raises a `ValueError` indicating that the dataset is unsupported.

4. **Dataset Loading:**
    * It creates two instances of the selected dataset class (`cls`): one for the training set (`train=True`) and one for the validation/test set (`train=False`).
    * For both instances, it specifies:
        * `root=data_dir`: The directory to save/load the data.
        * `download=True`: If the dataset is not found in the `data_dir`, it will be downloaded.
        * `transform=transform`: The defined sequence of transformations is applied to the images as they are loaded.

5. **DataLoader Creation:**
    * It creates two `DataLoader` instances using the loaded datasets: `train_loader` and `val_loader`.
    * `DataLoader` is a utility in PyTorch that helps in iterating over the dataset in batches.
    * For `train_loader`:
        * `dataset=train`: Uses the training dataset.
        * `batch_size=batch_size`: Sets the specified batch size.
        * `shuffle=True`: Shuffles the data in each epoch, which is important for training.
        * `num_workers=2`: Uses 2 subprocesses to load data, which can speed up data loading.
    * For `val_loader`:
        * `dataset=val`: Uses the validation/test dataset.
        * `batch_size=batch_size`: Sets the specified batch size.
        * `shuffle=False`: Does not shuffle the validation data, as shuffling is not necessary for evaluation.
        * `num_workers=2`: Uses 2 subprocesses for data loading.

6. **Return Values:** The function returns three values:
    * `train_loader`: The DataLoader for the training set.
    * `val_loader`: The DataLoader for the validation/test set.
    * `num_classes`: The number of classes in the dataset.

In summary, this function provides a convenient way to get ready-to-use data loaders and the number of classes for common image classification datasets, handling downloading, transformation, and batching automatically. It simplifies the process of loading and preparing data for machine learning tasks.

---

## Explanation of the `get_model` Function

This Python function, `get_model`, is designed to provide a flexible way to instantiate different neural network models based on a given name, number of output classes, and whether to use a pretrained version. It supports a custom LeNet5 implementation and leverages the `timm` library for a variety of other popular architectures.

Here's a breakdown of what it does:

1. **Function Definition:** It defines a function `get_model` that takes three arguments:
    * `name`: A string specifying the name of the model to load (e.g., 'lenet5', 'alexnet', 'resnet').
    * `num_classes`: An integer specifying the number of output classes for the model's final layer.
    * `pretrained`: A boolean indicating whether to load a model with weights pretrained on a large dataset (like ImageNet). Defaults to `False`.

2. **LeNet5 Implementation:**
    * It checks if the requested `name` is 'lenet5' (case-insensitive).
    * If it is, it defines a custom PyTorch `nn.Module` class called `LeNet5`.
    * The `LeNet5` class implements the classic LeNet-5 architecture, consisting of:
        * A `features` sequential module: Two convolutional layers (`nn.Conv2d`) followed by Tanh activation (`nn.Tanh`) and average pooling (`nn.AvgPool2d`).
        * A `classifier` sequential module: A flattening layer (`nn.Flatten`) to convert the 2D feature maps into a 1D vector, followed by three fully connected (linear) layers (`nn.Linear`) with Tanh activation between the first two, and a final linear layer mapping to `num_classes`.
    * The `forward` method defines how input data `x` flows through the network: first through the `features` and then through the `classifier`.
    * It returns an instance of this `LeNet5` class, assuming a single input channel (`in_ch=1`) by default. This is suitable for grayscale images like MNIST or FashionMNIST.

3. **timm Library Integration:**
    * If the name is not 'lenet5', the function checks if the name is one of the supported common architecture aliases defined in the `mapping` dictionary. This mapping translates user-friendly names ('alexnet', 'vggnet', etc.) to the specific model names used by the `timm` library.
    * If the name is found in the `mapping`, it uses `timm.create_model()` to instantiate the corresponding model from the `timm` library.
    * `timm.create_model()` is a powerful function that can load a wide variety of models. It takes the `model_name` (from the `mapping`), `pretrained` flag, and `num_classes` as arguments. `timm` handles adjusting the final classification layer to match the specified `num_classes`.

4. **Error Handling:**
    * If the provided `name` does not match 'lenet5' and is not found in the `mapping` for `timm` models, it raises a `ValueError` indicating that the model name is not recognized.

In summary, this function acts as a factory for creating neural network models, offering a custom LeNet5 and leveraging the extensive collection of models available in the `timm` library, with options for specifying the number of output classes and using pretrained weights.

## Training and Evaluation Functions

This code block contains two essential functions commonly used in a deep learning training pipeline, likely built with PyTorch: `train_one_epoch` for performing a single training pass over the dataset, and `evaluate` for assessing the model's performance on a validation or test set.

### `train_one_epoch` Function

This function handles the process of training a model for one full pass through the training dataset.

#### **Purpose:** 
To update the model's weights based on the training data using backpropagation

#### **Function Arguments:**

* **`model`**: The neural network model (`torch.nn.Module`).
* **`device`**: The device the model and data shouldbe on (e.g., 'cuda' or 'cpu').
* **`loader`**: A `DataLoader` providing batches oftraining data.
* **`criterion`**: The loss function (e.g., `nnCrossEntropyLoss`).
* **`optimizer`**: The optimization algorithm (e.g., `torch.optim.Adam`).

#### **Process:**

1. Sets the model to training mode (`model.train()`). This enables layers like Dropout and BatchNorm to behave appropriately during training.
2. Initializes a variable `total_loss` to accumulate the loss over the epoch.
3. Iterates through each batch of data provided by the `loader`. `tqdm` is used here to show a progress bar.
4. Moves the input features (`X`) and labels (`y`) to the specified `device`.
5. Resets the gradients of the optimizer to zero (`optimizer.zero_grad()`) before calculating gradients for the current batch.
6. Performs the forward pass: passes the input `X` through the `model` to get predictions.
7. Calculates the `loss` between the model's output and the true labels `y` using the `criterion`.
8. Performs the backward pass (`loss.backward()`) to compute the gradients of the loss with respect to the model's parameters.
9. Updates the model's weights using the computed gradients and the `optimizer` step (`optimizer.step()`).
10. Accumulates the batch loss, scaled by the number of samples in the batch (`X.size(0)`), into `total_loss`. `.item()` retrieves the loss as a standard Python number.

* **Return Value:** The average loss for the epoch, calculated as the total accumulated loss divided by the total number of samples in the dataset (`len(loader.dataset)`)

### `evaluate` Function

This function assesses the model's performance on a separate dataset (validation or test) without updating the model's weights.

#### **Purpose:** To measure the model's performance metrics like loss, accuracy, precision, recall, and F1-score on unseen data.

#### **Arguments:**

* **`model`**: The neural network model (`torch.nn.Module`).
* **`device`**: The device the model and data should be on (e.g., 'cuda' or 'cpu').
* **`loader`**: A `DataLoader` providing batches of evaluation data.
* **`criterion`**: The loss function (e.g., `nn.CrossEntropyLoss`).
* **`num_classes`**: The number of classes in the dataset.

#### **Process:**

1. Sets the model to evaluation mode (`model.eval()`). This disables layers like Dropout and sets BatchNorm to use running statistics.
2. Initializes `total_loss`, and empty lists `all_preds` and `all_targets` to store predictions and ground truth labels across all batches.
3. Uses `torch.no_grad()` context manager. This disables gradient calculation, which saves memory and computation as weights are not being updated.
4. Iterates through each batch of data provided by the `loader`. `tqdm` is used for a progress bar.
5. Moves the input features (`X`) and labels (`y`) to the specified `device`.
6. Performs the forward pass: passes the input `X` through the `model` to get `out`.
7. Accumulates the batch loss using the `criterion`, scaled by the number of samples in the batch, into `total_loss`.
8. Gets the predicted class for each sample in the batch using `out.argmax(dim=1)`.
9. Extends the `all_preds` and `all_targets` lists with the predictions and true labels from the current batch, moving them back to the CPU and converting to NumPy arrays.

#### **Return Value:** A tuple containing:

* `avg_loss`: The average loss over the evaluation dataset.
* `acc`: The overall accuracy.
* `prec`: The weighted average precision.
* `rec`: The weighted average recall.
* `f1`: The weighted average F1-score.
* `(all_targets, all_preds)`: A tuple containing lists of all true labels and all predictions across the entire dataset, useful for further analysis (e.g., confusion matrix).

---
## Explanation of the `plot_metrics` Function

This Python function, `plot_metrics`, is designed to visualize the training and validation progress of a machine learning model using `matplotlib`. It takes a history object (typically a dictionary containing metric values per epoch) and a title, then generates plots for loss and accuracy.

Here's a breakdown of what it does:

1. **Function Definition:** It defines a function `plot_metrics` that takes two arguments:
    * `history`: A dictionary expected to contain lists of metric values, specifically `'train_loss'`, `'val_loss'`, and `'val_acc'`, where each list contains values recorded at the end of each training epoch.
    * `title`: A string used as a base for the plot titles (e.g., the name of the model or experiment).

2. **Extracting Epoch Count:**
    * `epochs = len(history['train_loss'])`: It determines the number of epochs that were run by checking the length of the list storing the training loss values. This assumes that all metric lists in the `history` dictionary have the same length.

3. **Plotting Loss:**
    * `plt.figure()`: Creates a new figure for the loss plot.
    * `plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss')`: Plots the training loss against the epoch numbers (from 1 to `epochs`). The line is labeled 'Train Loss'.
    * `plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss')`: Plots the validation loss against the epoch numbers. The line is labeled 'Val Loss'.
    * `plt.title(f'{title} Loss')`: Sets the title of the plot using the provided `title` and appending ' Loss'.
    * `plt.xlabel('Epoch')`: Labels the x-axis as 'Epoch'.
    * `plt.legend()`: Displays the legend, showing which line corresponds to 'Train Loss' and 'Val Loss'.
    * `plt.show()`: Displays the generated loss plot.

4. **Plotting Accuracy:**
    * `plt.figure()`: Creates a *new* figure specifically for the accuracy plot.
    * `plt.plot(range(1, epochs+1), history['val_acc'], label='Val Acc')`: Plots the validation accuracy against the epoch numbers. The line is labeled 'Val Acc'. (Note: This function only plots validation accuracy, not training accuracy).
    * `plt.title(f'{title} Accuracy')`: Sets the title of this second plot using the provided `title` and appending ' Accuracy'.
    * `plt.xlabel('Epoch')`: Labels the x-axis as 'Epoch'.
    * `plt.legend()`: Displays the legend for the accuracy plot.
    * `plt.show()`: Displays the generated accuracy plot.

In summary, this function takes training history data, calculates the number of epochs, and then uses `matplotlib` to create and display two separate plots: one showing training and validation loss over epochs, and another showing validation accuracy over epochs.

---

# Explanation of the `run_experiment` Function

This Python function, `run_experiment`, orchestrates the entire process of training and evaluating a neural network model on a specified dataset. It brings together the data loading (`get_dataset`), model creation (`get_model`), training loop, evaluation, model saving, and results visualization/reporting.

Here's a breakdown of what it does:

1.  **Function Definition:** It defines the `run_experiment` function with several parameters, allowing customization of the experiment:
    * `dataset` (str): Name of the dataset to use (defaults to 'mnist').
    * `model_name` (str): Name of the model architecture to use (defaults to 'lenet5').
    * `pretrained` (bool): Whether to use a pretrained model (defaults to `False`).
    * `epochs` (int): The number of training epochs (defaults to 5).
    * `batch_size` (int): The batch size for data loaders (defaults to 64).
    * `lr` (float): The learning rate for the optimizer (defaults to 0.01).

2.  **Device Configuration:**
    * It determines the device to use for training and evaluation. It checks if a CUDA-enabled GPU is available (`torch.cuda.is_available()`) and sets the device to 'cuda' if it is, otherwise it uses the 'cpu'.

3.  **Data Loading:**
    * It calls the `get_dataset` function (presumably defined elsewhere) with the specified `dataset` name and `batch_size`.
    * It receives the `train_loader`, `val_loader`, and `num_classes` from `get_dataset`.
    * It then inspects the shape of the first batch from the `train_loader` to determine the number of input channels (`in_ch`) for the model. This is important because different datasets (like MNIST/FashionMNIST vs. CIFAR10) have different numbers of channels (1 vs. 3).

4.  **Model Creation:**
    * It calls the `get_model` function (presumably defined elsewhere) with the specified `model_name`, `num_classes`, and `pretrained` flag.
    * It receives the instantiated model.
    * **LeNet5 Input Channel Adaptation:** If the selected model is 'lenet5', it specifically modifies the first convolutional layer (`model.features[0]`) to match the determined number of input channels (`in_ch`). This is necessary because the custom LeNet5 implementation in `get_model` defaults to 1 input channel, but datasets like CIFAR10 have 3.
    * It moves the model to the selected device (`model.to(device)`).

5.  **Loss Function and Optimizer:**
    * It defines the criterion (loss function) for training. `nn.CrossEntropyLoss()` is used, which is standard for multi-class classification.
    * It defines the optimizer. `optim.SGD` (Stochastic Gradient Descent) with momentum is used to update the model's weights based on the calculated gradients.

6.  **Training and Evaluation Loop:**
    * It initializes a dictionary `history` to store metrics (train loss, validation loss, validation accuracy) for each epoch.
    * It initializes `best_acc` to 0.0 to keep track of the highest validation accuracy achieved.
    * It iterates through the specified number of `epochs`.
    * Inside the loop:
        * It prints the current epoch number.
        * It calls `train_one_epoch` (presumably defined elsewhere) to train the model for one epoch using the `train_loader`. It receives the training loss (`tl`).
        * It calls `evaluate` (presumably defined elsewhere) to evaluate the model on the `val_loader`. It receives various metrics including validation loss (`vl`), validation accuracy (`va`), precision (`vp`), recall (`vr`), F1 score (`vf1`), and the true and predicted labels.
        * It appends the calculated metrics to the `history` dictionary.
        * It prints the training loss, validation loss, validation accuracy, and F1 score for the current epoch.
        * **Model Saving:** It checks if the current validation accuracy (`va`) is greater than the `best_acc` seen so far. If it is, it updates `best_acc` and saves the model's state dictionary (`model.state_dict()`) to a file named based on the dataset and model name (e.g., 'mnist_lenet5_best.pth'). This saves the weights of the best performing model on the validation set.

7.  **Results Visualization and Reporting:**
    * After the training loop finishes, it calls `plot_metrics` (presumably defined elsewhere) to visualize the training and validation metrics stored in the `history`.
    * It performs a final evaluation on the `val_loader` to get the true and predicted labels.
    * It prints a detailed classification report using `classification_report` from `sklearn.metrics`, showing precision, recall, F1-score, and support for each class.

8.  **Return Value:**
    * The function returns the trained model instance.

In essence, `run_experiment` encapsulates a standard deep learning training workflow, making it easy to run experiments with different datasets, models, and hyperparameters. It also includes mechanisms for saving the best model based on validation performance and visualizing training and validation metrics.