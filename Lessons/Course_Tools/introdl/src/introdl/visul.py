import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, SelectMultiple, Dropdown, Button, HBox, Text, VBox, Output, Layout, RadioButtons, IntSlider
from ipycanvas import Canvas
from IPython.display import display, clear_output
import torchvision.utils as vutils
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.pyplot import get_cmap
import matplotlib.gridspec as gridspec
from scipy.special import erf
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torchvision.transforms as transforms
from typing import Union, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from introdl.utils import classifier_predict

########################################################
# Visualization Related Functions
########################################################

# === REMOVED_FUNCTION_START: in_notebook ===
# def in_notebook():
#     """
#     Check if the code is running in a Jupyter notebook environment.

#     Returns:
#         bool: True if running in a Jupyter notebook, False otherwise.
#     """
#     try:
#         shell = get_ipython().__class__.__name__
#         return shell == 'ZMQInteractiveShell'  # Indicates Jupyter notebook or JupyterLab
#     except NameError:
#         return False  # Not in a notebook environment
# === REMOVED_FUNCTION_END: in_notebook ===


def image_to_PIL(image, cmap=None, mean=None, std=None):
    """
    Preprocess an image to numpy RGB format.
    Parameters:
    image: The input image to preprocess. It can be a PyTorch tensor, a PIL image, or a numpy array.
    cmap: The colormap to apply if the image is grayscale. Default is None.
    mean: The mean values for each channel for denormalization. Default is None.
    std: The standard deviation values for each channel for denormalization. Default is None.
    
    Returns:
    Image.Image: The preprocessed image in RGB format as a PIL image.
    Raises:
    TypeError: If the input image type is unsupported.
    ValueError: If the input image has an unexpected shape.
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Handle (C, H, W)
            image = np.transpose(image, (1, 2, 0))
    elif isinstance(image, Image.Image):  # Convert PIL image to numpy
        image = np.array(image)
    elif not isinstance(image, np.ndarray):  # Ensure image is numpy
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Handle grayscale detection
    if image.ndim == 2:  # Single-channel grayscale
        is_grayscale = True
    elif image.ndim == 3 and image.shape[2] == 1:  # 1-channel grayscale
        image = np.squeeze(image, axis=2)
        is_grayscale = True
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        is_grayscale = np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(image[:, :, 1], image[:, :, 2])
        if is_grayscale:
            image = image[:, :, 0]  # Collapse to single channel
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Denormalize if mean and std are provided
    if mean is not None and std is not None:
        mean = np.array(mean).reshape(1, 1, -1)
        std = np.array(std).reshape(1, 1, -1)
        if is_grayscale:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)
        image = image * std + mean

    # Apply colormap if grayscale
    if cmap and is_grayscale:
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        colormap = plt.get_cmap(cmap)
        image = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
    elif is_grayscale:  # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)

    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    return Image.fromarray(image)

def create_image_grid(dataset, nrows, ncols, img_size=(64, 64), padding=2, label_height=12, 
                      class_labels=None, indices=None, show_labels=False, dark_mode=False, cmap=None,
                      mean=None, std=None, pad=False):
    """
    Creates a grid of images from a given dataset.

    Creates a visually appealing grid layout of images from a PyTorch dataset for data
    exploration and visualization. Handles normalization, class labels, and formatting.
    Used extensively in L02-L05 for visualizing datasets and model predictions.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing images and labels.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        img_size (tuple, optional): Size of each image (height, width). Defaults to (64, 64).
        padding (int, optional): Padding between images. Defaults to 2.
        label_height (int, optional): Height of label area below images. Defaults to 12.
        class_labels (list, optional): Class label names. If None, uses dataset classes. Defaults to None.
        indices (array, optional): Specific indices to show. If None, random selection. Defaults to None.
        show_labels (bool, optional): Whether to show labels below images. Defaults to False.
        dark_mode (bool, optional): Use dark theme. Defaults to False.
        cmap (str, optional): Colormap for grayscale images. Defaults to None.
        mean (tuple, optional): Mean for denormalization. Defaults to None.
        std (tuple, optional): Std for denormalization. Defaults to None.
        pad (bool, optional): Add padding around grid. Defaults to False.

    Returns:
        None: Displays the image grid using matplotlib.

    Example:
        Explore MNIST dataset (L02_1_MNIST_FC):

        ```python
        from introdl.visul import create_image_grid

        # Show random sample of MNIST digits
        create_image_grid(train_dataset, nrows=4, ncols=8,
                         show_labels=True, img_size=(32, 32))

        # Show specific examples with class names
        mnist_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        create_image_grid(test_dataset, nrows=2, ncols=10,
                         class_labels=mnist_labels,
                         show_labels=True)
        ```

        Visualize CIFAR10 (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Explore CIFAR10 classes
        cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        create_image_grid(train_dataset, nrows=3, ncols=10,
                         class_labels=cifar_labels,
                         show_labels=True,
                         img_size=(48, 48))
        ```

        Compare predictions (L02_3_MNIST_CNN):
        ```python
        # Show model predictions vs ground truth
        create_image_grid(test_dataset, nrows=2, ncols=5,
                         indices=misclassified_indices,
                         show_labels=True,
                         class_labels=mnist_labels)
        ```

        Custom visualization (L05_1_Transfer_Learning):
        ```python
        # High-resolution grid for transfer learning dataset
        create_image_grid(flowers_dataset, nrows=3, ncols=4,
                         img_size=(128, 128),
                         show_labels=True,
                         padding=5)
        ```
    """
    if class_labels is None and hasattr(dataset, 'classes'):
        class_labels = dataset.classes

    if indices is None:
        indices = np.random.choice(len(dataset), min(len(dataset), nrows * ncols), replace=False)

    # Calculate the actual number of rows and columns needed
    actual_nrows = (len(indices) + ncols - 1) // ncols
    actual_nrows = min(actual_nrows, nrows)
    actual_ncols = min(ncols, len(indices))

    # Calculate canvas size
    img_width, img_height = img_size
    canvas_width = actual_ncols * img_width + (actual_ncols - 1) * padding
    canvas_height = actual_nrows * (img_height + (label_height if show_labels else 0)) + (actual_nrows - 1) * padding

    # Create blank canvas with white or black background
    bg_color = (0, 0, 0) if dark_mode else (255, 255, 255)
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Default font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # Place each image and label on the canvas
    for idx, data_idx in enumerate(indices):
        image, label = dataset[data_idx]
        image = image_to_PIL(image, cmap=cmap, mean=mean, std=std)  # Preprocess the image

        if pad:
            image = ImageOps.pad(image, img_size, color=bg_color)
        else:
            image = image.resize(img_size, Image.Resampling.LANCZOS)

        row, col = divmod(idx, actual_ncols)
        x = col * (img_width + padding)
        y = row * (img_height + (label_height if show_labels else 0) + padding)

        canvas.paste(image, (x, y + (label_height if show_labels else 0)))

        if show_labels:
            label_text = class_labels[label] if class_labels else f'{label}'
            text_color = (255, 255, 255) if dark_mode else (0, 0, 0)
            text_x = x + img_width // 2
            text_y = y + label_height // 2
            draw.text((text_x, text_y), label_text, fill=text_color, font=font, anchor="mm")

    '''
    # Display the final grid image
    if in_notebook():
        display(canvas)  # Display inline in Jupyter notebook
    else:
        canvas.show()  # Open in a separate window if not in a notebook
    '''
    display(canvas) # Display inline in Jupyter notebook

def plot_transformed_images(dataset, num_images=5, num_transformed=5, img_size=(64, 64), padding=2, dark_mode=False, cmap=None, mean=None, std=None):
    """
    Plots a grid of randomly sampled images from the dataset along with their transformed versions.

    Visualizes the effect of data augmentation by showing original images alongside
    their randomly transformed versions. Essential for understanding data augmentation
    effects and verifying transform pipelines work correctly. Used in L03 and L05
    to demonstrate augmentation techniques.

    Args:
        dataset (torch.utils.data.Dataset): Dataset with transforms applied.
        num_images (int): Number of original images to sample. Defaults to 5.
        num_transformed (int): Number of transformed versions per image. Defaults to 5.
        img_size (tuple, optional): Size of each image (height, width). Defaults to (64, 64).
        padding (int, optional): Padding between images. Defaults to 2.
        dark_mode (bool, optional): Use dark background. Defaults to False.
        cmap (str, optional): Colormap for grayscale images. Defaults to None.
        mean (tuple, optional): Mean for denormalization. Defaults to None.
        std (tuple, optional): Standard deviation for denormalization. Defaults to None.

    Returns:
        None: Displays grid showing original and augmented versions.

    Example:
        Visualize data augmentation (L03_3_Augmentation_with_CIFAR10):

        ```python
        from introdl.visul import plot_transformed_images

        # Create dataset with augmentation
        train_loader, test_loader = create_CIFAR10_loaders(
            use_augmentation=True,
            data_dir=DATA_PATH
        )

        # Show augmentation effects
        plot_transformed_images(
            train_loader.dataset,
            num_images=4,
            num_transformed=6,
            img_size=(48, 48),
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
        ```

        Transfer learning dataset (L05_1_Transfer_Learning):
        ```python
        # Visualize augmented transfer learning data
        plot_transformed_images(
            augmented_flowers_dataset,
            num_images=3,
            num_transformed=5,
            img_size=(128, 128),
            padding=3
        )
        ```

        Compare augmentation strategies (L03_2_Resizing_and_Augmentation):
        ```python
        # Show different augmentation intensities
        mild_dataset = create_dataset_with_mild_augmentation()
        aggressive_dataset = create_dataset_with_aggressive_augmentation()

        # Compare side by side
        plot_transformed_images(mild_dataset, num_images=3)
        plot_transformed_images(aggressive_dataset, num_images=3)
        ```
    """
    indices = np.random.choice(len(dataset), num_images, replace=False)

    # Calculate canvas size
    img_width, img_height = img_size
    canvas_width = (num_transformed + 1) * img_width + num_transformed * padding
    canvas_height = num_images * img_height + (num_images - 1) * padding

    # Create blank canvas with white or black background
    bg_color = (0, 0, 0) if dark_mode else (255, 255, 255)
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    for i, idx in enumerate(indices):
        original_image, _ = dataset[idx]
        original_image = image_to_PIL(original_image, cmap=cmap, mean=mean, std=std)
        original_image = original_image.resize(img_size, Image.Resampling.LANCZOS)

        x = 0
        y = i * (img_height + padding)
        canvas.paste(original_image, (x, y))

        for j in range(1, num_transformed + 1):
            transformed_image, _ = dataset[idx]
            transformed_image = image_to_PIL(transformed_image, cmap=cmap, mean=mean, std=std)
            transformed_image = transformed_image.resize(img_size, Image.Resampling.LANCZOS)

            x = j * (img_width + padding)
            canvas.paste(transformed_image, (x, y))

    '''
    # Display the final grid image
    if in_notebook():
        display(canvas)  # Display inline in Jupyter notebook
    else:
        canvas.show()  # Open in a separate window if not in a notebook
    '''
    display(canvas) # Display inline in Jupyter notebook


def evaluate_classifier(model, dataset, device, display_confusion=True, img_size=(5, 5), 
                        batch_size=32, use_class_labels=True):
    """
    Evaluates the model on the given dataset, plots the confusion matrix if specified,
    and returns the confusion matrix, classification report, and misclassified dataset.

    Comprehensive model evaluation function that provides multiple analysis tools.
    Creates confusion matrix visualization, classification report with metrics,
    and identifies misclassified examples for further analysis. Essential for
    understanding model performance in classification tasks.

    Args:
        model (torch.nn.Module): The trained classification model to evaluate.
        dataset (torch.utils.data.Dataset): The dataset to evaluate on (typically test set).
        device (torch.device): The device to run evaluation on.
        display_confusion (bool, optional): Whether to display confusion matrix. Defaults to True.
        img_size (tuple, optional): Size of confusion matrix plot. Defaults to (5, 5).
        batch_size (int, optional): Batch size for evaluation. Defaults to 32.
        use_class_labels (bool, optional): Use class labels in confusion matrix. Defaults to True.

    Returns:
        tuple: (confusion_matrix, classification_report, misclassified_dataset)
            - confusion_matrix: numpy array of prediction vs true label counts
            - classification_report: dict with precision, recall, f1-score per class
            - misclassified_dataset: dataset containing only misclassified examples

    Example:
        Evaluate MNIST CNN (L02_1_MNIST_FC):

        ```python
        from introdl.visul import evaluate_classifier

        # Load trained model
        model = load_model(SimpleCNN, MODELS_PATH / 'L02_MNIST_CNN.pt')

        # Comprehensive evaluation
        conf_matrix, class_report, misclassified = evaluate_classifier(
            model, test_dataset, device,
            display_confusion=True,
            img_size=(8, 6)
        )

        # Analyze results
        print(f"Overall accuracy: {class_report['accuracy']:.3f}")
        print(f"Misclassified examples: {len(misclassified)}")
        ```

        Detailed analysis (L02_3_MNIST_CNN):
        ```python
        # Evaluate with detailed class analysis
        conf_matrix, report, misclassified = evaluate_classifier(
            model, test_dataset, device
        )

        # Find worst performing classes
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                f1 = metrics['f1-score']
                if f1 < 0.90:
                    print(f"Class '{class_name}' needs attention: F1={f1:.3f}")

        # Visualize misclassified examples
        if len(misclassified) > 0:
            create_image_grid(misclassified, nrows=2, ncols=5,
                             show_labels=True, img_size=(64, 64))
        ```

        CIFAR10 evaluation (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Evaluate CIFAR10 model
        cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        conf_matrix, report, misclassified = evaluate_classifier(
            model, test_dataset, device,
            display_confusion=True
        )

        # Check class-specific performance
        difficult_classes = []
        for i, class_name in enumerate(cifar_labels):
            class_accuracy = conf_matrix[i, i] / conf_matrix[i, :].sum()
            if class_accuracy < 0.7:
                difficult_classes.append(class_name)

        print(f"Difficult classes: {difficult_classes}")
        ```
    """
    # Get predictions and labels using the classifier_predict function
    pred_labels, labels = classifier_predict(dataset, model, device, return_labels=True, batch_size=batch_size)

    # Collect misclassified indices
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(labels, pred_labels)) if true != pred]

    print(f'The dataset has {len(dataset)} samples.')
    print(f'The model misclassified {len(misclassified_indices)} samples.')

    # Check to see if classes are available in dataset.classes or dataset.dataset.classes
    if hasattr(dataset, 'classes') and use_class_labels:
        classes = dataset.classes
    elif hasattr(dataset, 'dataset.classes') and use_class_labels:
        classes = dataset.dataset.classes
    else:
        classes = None


    # Create a list of strings in the form "truth / predict" for each misclassified index
    misclassified_labels = []
    for idx in misclassified_indices:
        true_label = classes[labels[idx]] if classes else labels[idx]
        pred_label = classes[pred_labels[idx]] if classes else pred_labels[idx]
        misclassified_labels.append(f'{true_label} / {pred_label}')

    # Create a new dataset containing only the misclassified images
    class MisclassifiedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, misclassified_indices, misclassified_labels):
            self.dataset = dataset
            self.misclassified_indices = misclassified_indices
            self.misclassified_labels = misclassified_labels

        def __len__(self):
            return len(self.misclassified_indices)

        def __getitem__(self, idx):
            original_idx = self.misclassified_indices[idx]
            image, _ = self.dataset[original_idx]
            label = self.misclassified_labels[idx]
            return image, label

    misclassified_dataset = MisclassifiedDataset(dataset, misclassified_indices, misclassified_labels)

    # Compute the confusion matrix
    confusion_mat = confusion_matrix(labels, pred_labels)

    # Display the confusion matrix if specified
    if display_confusion:
        fig, ax = plt.subplots(figsize=img_size)
        disp = ConfusionMatrixDisplay(confusion_mat, 
                                      display_labels=classes if classes else None)
        disp.plot(ax=ax)
        # Rotate x-axis labels
        if classes:
            ax.set_xticklabels(classes, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Generate the classification report
    class_report = classification_report(labels, pred_labels, 
                                         target_names=classes if classes else None)

    return confusion_mat, class_report, misclassified_dataset

def visualize2DSoftmax(X, y, model, title=None, numpts=20):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=numpts), np.linspace(y_min, y_max, num=numpts), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    plt.figure(figsize=(5,5))
    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(numpts,numpts), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)
  
# Function to retrieve filters for all convolutional layers
def _get_conv_filters(model):
    """
    Retrieves the filters (weights) for each convolutional layer in the model.

    Args:
        model (torch.nn.Module): The trained CNN model.

    Returns:
        List[torch.Tensor]: A list of filter tensors for each convolutional layer.
    """
    conv_filters = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_filters.append(layer.weight.detach().cpu())
    return conv_filters

# Function to normalize the filters for visualization
def _normalize_filters(filters):
    """
    Normalizes the filters to the range [0, 1] for better visualization.
    
    Args:
        filters (torch.Tensor): The filters tensor.

    Returns:
        torch.Tensor: The normalized filters tensor.
    """
    min_val = filters.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = filters.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return (filters - min_val) / (max_val - min_val)

# Function to visualize the filters of a specified convolutional layer
# === REMOVED_FUNCTION_START: visualize_filters ===
# def visualize_filters(model, layer_idx=1, padding=1, scale=1.0):
#     """
#     Visualizes the filters from a specified convolutional layer in the model.

#     Args:
#         model (torch.nn.Module): The trained CNN model.
#         layer_idx (int, optional): The index of the convolutional layer to visualize (1-based indexing). Defaults to 1.
#         padding (int, optional): Padding between filters in the grid. Defaults to 1.
#         scale (float, optional): Scaling factor for the figure size. Defaults to 1.0.
#     """
#     conv_filters = _get_conv_filters(model)
#     layer_idx = layer_idx - 1  # Convert to 0-based index
    
#     if layer_idx >= len(conv_filters) or layer_idx < 0:
#         raise ValueError(f"Layer index out of range. The model has {len(conv_filters)} convolutional layers.")
    
#     filters = conv_filters[layer_idx]
#     num_filters = filters.shape[0]
#     in_channels = filters.shape[1]
#     kernel_size = filters.shape[2:]

#     # Normalize filters for better visualization
#     filters = _normalize_filters(filters)

#     # Determine the number of rows and columns for the subplot grid
#     fig, axs = plt.subplots(num_filters, in_channels, figsize=(in_channels * 2, num_filters * 2))

#     for i in range(num_filters):
#         for j in range(in_channels):
#             # Handle single row or single column cases
#             if num_filters == 1 and in_channels == 1:
#                 ax = axs
#             elif num_filters == 1:
#                 ax = axs[j]
#             elif in_channels == 1:
#                 ax = axs[i]
#             else:
#                 ax = axs[i, j]
            
#             ax.imshow(filters[i, j], cmap='gray', aspect='auto')
#             ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()
# === REMOVED_FUNCTION_END: visualize_filters ===


################ Plotting Activation Functions ################

activation_functions = {
    "sigmoid": lambda x: (1 / (1 + np.exp(-x)), lambda y: y * (1 - y)),
    "ReLU": lambda x: (np.maximum(0, x), lambda y: np.where(x > 0, 1, 0)),
    "tanh": lambda x: (np.tanh(x), lambda y: 1 - y ** 2),
    "LeakyReLU": lambda x: (np.where(x > 0, x, 0.1 * x), lambda y: np.where(x > 0, 1, 0.1 * np.ones_like(x))),
    "GELU": lambda x: (x * 0.5 * (1 + erf(x / np.sqrt(2))), lambda y: 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-0.5 * x ** 2)) / np.sqrt(2 * np.pi)),
    "swish": lambda x: (x / (1 + np.exp(-x)), lambda y: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x)) ** 2),
}

# Define the function to plot selected activation functions
# === REMOVED_FUNCTION_START: plot_activation_functions ===
# def plot_activation_functions(selected_activations):
#     """
#     Plots the activation functions and their derivatives.

#     Parameters:
#     - selected_activations (str or list): A string or a list of activation function names.
#         Possible activation function names: 'sigmoid', 'ReLU', 'tanh', 'LeakyReLU', 'GELU', 'swish'

#     Returns:
#     None
#     """
#     x = np.linspace(-6, 6, 100)
#     data = {
#         "x": [],
#         "value": [],
#         "derivative": [],
#         "activation": [],
#     }

#     if isinstance(selected_activations, str):
#         selected_activations = [selected_activations]

#     for activation in selected_activations:
#         if activation not in activation_functions.keys():
#             raise ValueError(f"Invalid activation function: {activation}")
#         func = activation_functions[activation]
#         y, dy_func = func(x)
#         dy = dy_func(y)

#         # Append data for plotting
#         data["x"].extend(x)
#         data["value"].extend(y)
#         data["derivative"].extend(dy)
#         data["activation"].extend([activation] * len(x))

#     # Create DataFrame from the collected data
#     df = pd.DataFrame(data)

#     # Set up the plotting style and layout
#     sns.set(style="whitegrid", palette="muted")
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # Adjusted aspect ratio

#     # Plot activation functions
#     sns.lineplot(
#         data=df, 
#         x="x", 
#         y="value", 
#         hue="activation", 
#         style="activation", 
#         ax=axes[0]
#     )
#     axes[0].set_title("Activation Functions")
#     axes[0].legend(title='Activation')

#     # Plot derivatives of activation functions
#     sns.lineplot(
#         data=df, 
#         x="x", 
#         y="derivative", 
#         hue="activation", 
#         style="activation", 
#         ax=axes[1]
#     )
#     axes[1].set_title("Derivatives")
#     axes[1].legend(title="Activation'")

#     plt.show()
# === REMOVED_FUNCTION_END: plot_activation_functions ===


# === REMOVED_FUNCTION_START: activations_widget ===
# def activations_widget():

#     # Create a widget for selecting activation functions
#     activation_selector = widgets.SelectMultiple(
#         options=list(activation_functions.keys()),
#         value=["sigmoid"],  # Default value
#         description="Activations",
#         disabled=False
#     )

#     # Use the interactive function to connect the selector widget to the plot function
#     interactive_plot = interactive(plot_activation_functions, selected_activations=activation_selector)
#     output = interactive_plot.children[-1]
#     output.layout.height = '500px'
#     display(interactive_plot)
# === REMOVED_FUNCTION_END: activations_widget ===



########################################################
# Feauture Map Visualization Related Functions
########################################################

def vis_feature_maps(dataset, model, target_index, mean, std, layer=1, activation=None, pooling=None, original_image_size=(5, 5), feature_maps_size=(10, 10), cmap_name='PuOr'):
    """
    Visualizes the feature maps from a CNN model for a target image in the dataset using GridSpec to control the layout.

    Shows what CNN filters detect by visualizing feature map activations for a specific input image.
    Essential for understanding how CNNs process images and what features each layer learns.
    Used in L02-L03 to explain convolutional operations and feature learning.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        model (torch.nn.Module): The trained CNN model.
        target_index (int): The index of the target image in the dataset.
        mean (tuple): The mean values for denormalization (e.g., CIFAR10: (0.4914, 0.4822, 0.4465)).
        std (tuple): The standard deviation values for denormalization (e.g., CIFAR10: (0.2023, 0.1994, 0.2010)).
        layer (int, optional): The convolutional layer index to visualize. Defaults to 1.
        activation (str, optional): Activation to apply ('relu', 'tanh', None). Defaults to None.
        pooling (str, optional): Pooling to apply ('max', 'average', None). Defaults to None.
        original_image_size (tuple, optional): Figure size for original image. Defaults to (5, 5).
        feature_maps_size (tuple, optional): Figure size for feature maps. Defaults to (10, 10).
        cmap_name (str, optional): Colormap for feature maps. Defaults to 'PuOr'.

    Returns:
        None: Displays the original image and feature maps side by side.

    Example:
        Visualize CNN features (L02_3_MNIST_CNN):

        ```python
        from introdl.visul import vis_feature_maps

        # Load trained CNN model
        model = load_model(SimpleCNN, MODELS_PATH / 'L02_MNIST_CNN.pt')

        # Visualize what first layer detects
        vis_feature_maps(
            dataset=test_dataset,
            model=model,
            target_index=42,  # Specific image to analyze
            mean=(0.1307,),   # MNIST normalization
            std=(0.3081,),
            layer=1,          # First conv layer
            original_image_size=(4, 4),
            feature_maps_size=(12, 8)
        )
        ```

        Compare layers (L02_3_MNIST_CNN):
        ```python
        # Compare different layers' feature maps
        for layer_idx in [1, 2]:
            print(f"Layer {layer_idx} feature maps:")
            vis_feature_maps(test_dataset, model, 0,
                           mean=(0.1307,), std=(0.3081,),
                           layer=layer_idx,
                           feature_maps_size=(15, 10))
        ```

        CIFAR10 feature visualization (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Visualize CIFAR10 CNN features
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)

        vis_feature_maps(
            dataset=test_dataset,
            model=model,
            target_index=100,
            mean=cifar_mean,
            std=cifar_std,
            layer=1,
            activation='relu',  # Show ReLU activation effect
            original_image_size=(6, 6),
            feature_maps_size=(12, 12)
        )
        ```

        Advanced analysis (L04_1_Advanced_NN):
        ```python
        # Analyze deeper layer features
        vis_feature_maps(test_dataset, model, 25,
                       mean=cifar_mean, std=cifar_std,
                       layer=3,  # Deeper layer shows more complex features
                       pooling='max',  # Apply max pooling
                       cmap_name='viridis')
        ```
    """
    model.eval()
    device = next(model.parameters()).device
    image, label = dataset[target_index]
    image = image.unsqueeze(0).to(device)

    # Hook function to extract feature maps
    conv_outputs = []
    
    def hook_fn(module, input, output):
        conv_outputs.append(output)

    # Register hooks to all convolutional layers
    conv_layers = [layer_module for layer_module in model.modules() if isinstance(layer_module, torch.nn.Conv2d)]
    if layer > len(conv_layers):
        raise ValueError(f"The model has only {len(conv_layers)} convolutional layers, but layer {layer} was requested.")

    # Attach the hook to the specific layer
    hooks = [conv_layers[layer - 1].register_forward_hook(hook_fn)]

    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not conv_outputs:
        raise RuntimeError(f"No feature maps were captured. Make sure the requested layer {layer} is valid.")

    # Get the feature maps from the specified layer
    feature_maps = conv_outputs[0].squeeze(0)  # Get the feature maps for the layer

    # Apply the requested activation function
    if activation == 'relu':
        feature_maps = torch.relu(feature_maps)
    elif activation == 'tanh':
        feature_maps = torch.tanh(feature_maps)

    # Apply pooling if requested
    if pooling == 'max':
        feature_maps = torch.nn.functional.max_pool2d(feature_maps, kernel_size=2, stride=2)
    elif pooling == 'average':
        feature_maps = torch.nn.functional.avg_pool2d(feature_maps, kernel_size=2, stride=2)

    # Normalize feature maps to [0, 1] range for consistent visualization
    abs_max = torch.max(torch.abs(feature_maps)).item()
    vmin, vmax = -abs_max, abs_max

    # Apply colormap to feature maps
    def apply_colormap(feature_map, cmap_name):
        feature_map_np = feature_map.cpu().numpy()
        cmap = get_cmap(cmap_name)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        colored_map = cmap(norm(feature_map_np))[:, :, :3]  # Apply colormap and remove alpha channel
        return torch.tensor(colored_map).permute(2, 0, 1)  # Convert to (3, H, W)

    colored_feature_maps = [apply_colormap(feature_maps[i], cmap_name) for i in range(feature_maps.shape[0])]

    # Create the grid of feature maps for display
    grid_image = vutils.make_grid(colored_feature_maps, nrow=int(len(colored_feature_maps) ** 0.5), padding=1, normalize=False)

    # Denormalize the original image for display
    def denormalize_image(image_tensor, mean, std):
        num_channels = image_tensor.shape[0]  # Get the number of channels (1 for MNIST, 3 for RGB)
        mean = torch.tensor(mean).view(num_channels, 1, 1).to(image_tensor.device)
        std = torch.tensor(std).view(num_channels, 1, 1).to(image_tensor.device)
        return image_tensor * std + mean

    original_image = denormalize_image(image.squeeze(0).cpu(), mean, std)

    # Calculate the relative sizes of the original image and feature maps
    original_image_width, original_image_height = original_image_size
    feature_maps_width, feature_maps_height = feature_maps_size

    total_width = original_image_width + feature_maps_width
    total_height = max(original_image_height, feature_maps_height)

    # Create a GridSpec layout with relative sizes
    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[original_image_width, feature_maps_width])

    # Plot the original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_image.permute(1, 2, 0).clip(0, 1) if original_image.shape[0] == 3 else original_image.squeeze(0), cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Original Image", fontsize=14)

    # Plot the feature maps
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(grid_image.permute(1, 2, 0).cpu().numpy())
    ax2.axis('off')
    ax2.set_title(f"Feature Maps for Layer {layer}", fontsize=14)

    plt.tight_layout()
    plt.show()


def vis_feature_maps_widget(model, dataset, initial_target_index=0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), original_image_size=(5, 5), feature_maps_size=(10, 10)):
    """
    Creates and displays an interactive widget for visualizing feature maps with the ability to adjust
    layer index, activation type, pooling type, and figure sizes for original image and feature maps.

    Interactive version of vis_feature_maps that allows real-time exploration of CNN
    feature maps across different layers, images, and processing options. Essential
    for understanding how CNNs learn hierarchical features. Used in L02-L03 for
    interactive CNN exploration.

    Args:
        model (torch.nn.Module): The trained CNN model to explore.
        dataset (torch.utils.data.Dataset): Dataset containing the images.
        initial_target_index (int, optional): Initial image index to display. Defaults to 0.
        mean (tuple, optional): Mean for denormalization. Defaults to ImageNet values.
        std (tuple, optional): Std for denormalization. Defaults to ImageNet values.
        original_image_size (tuple, optional): Size for original image plot. Defaults to (5, 5).
        feature_maps_size (tuple, optional): Size for feature maps plot. Defaults to (10, 10).

    Returns:
        None: Displays interactive widget with controls and visualization.

    Example:
        Interactive CNN exploration (L02_3_MNIST_CNN):

        ```python
        from introdl.visul import vis_feature_maps_widget

        # Load trained CNN model
        model = load_model(SimpleCNN, MODELS_PATH / 'L02_MNIST_CNN.pt')

        # Create interactive feature map explorer
        vis_feature_maps_widget(
            model=model,
            dataset=test_dataset,
            initial_target_index=42,
            mean=(0.1307,),  # MNIST normalization
            std=(0.3081,),
            original_image_size=(4, 4),
            feature_maps_size=(12, 8)
        )

        # Interactive controls allow students to:
        # - Change image index to see different examples
        # - Switch between conv layers (Layer 1, 2, 3, etc.)
        # - Apply different activations (None, ReLU, Tanh)
        # - Try different pooling (None, Max, Average)
        ```

        CIFAR10 exploration (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Explore CIFAR10 CNN features interactively
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)

        vis_feature_maps_widget(
            model=cifar_model,
            dataset=test_dataset,
            mean=cifar_mean,
            std=cifar_std,
            original_image_size=(6, 6),
            feature_maps_size=(14, 10)
        )

        # Students can explore:
        # - How different objects activate different filters
        # - Progression from edge detection to object features
        # - Effect of pooling on feature map resolution
        ```

        Compare architectures (L04_1_Advanced_NN):
        ```python
        # Compare basic vs advanced CNN feature learning
        print("Basic CNN features:")
        vis_feature_maps_widget(basic_model, dataset)

        print("Advanced CNN features:")
        vis_feature_maps_widget(advanced_model, dataset)

        # Students can compare:
        # - Feature complexity between architectures
        # - Number of filters learned
        # - Quality of feature representations
        ```
    """
    # Extract the convolutional layers from the model
    conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    num_conv_layers = len(conv_layers)
    
    if num_conv_layers == 0:
        raise ValueError("The model does not contain any convolutional layers.")

    # Adjust slider width based on the number of layers
    slider_width = '200px' if num_conv_layers <= 2 else '400px'

    def update_visualization(change=None):
        """
        Function to update the visualization based on widget values.
        Called when any widget value changes.
        """
        clear_output(wait=True)
        display(ui_and_visualization)  # Display the UI and visualization container again
        vis_feature_maps(
            dataset=dataset,
            model=model,
            target_index=target_index_widget.value,
            mean=mean,
            std=std,
            layer=layer_widget.value,
            activation=activation_widget.value,
            pooling=pooling_widget.value,
            original_image_size=original_image_size,  # Pass custom size for the original image
            feature_maps_size=feature_maps_size  # Pass custom size for the feature maps
        )

    # Define a fixed width for the entire UI and visualization
    fixed_width = '800px'  # Adjust this width as necessary to match your desired size

    # Widgets for interactive control
    layer_widget = widgets.IntSlider(
        value=1,  # Default layer index
        min=1,    # Minimum layer index
        max=num_conv_layers,  # Set max to the number of convolutional layers
        step=1,
        description='Layer Index',
        layout=widgets.Layout(width=slider_width)  # Adjust slider width based on number of layers
    )

    activation_widget = widgets.Dropdown(
        options=[None, 'relu', 'tanh'],
        value=None,
        description='Activation',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    pooling_widget = widgets.Dropdown(
        options=[None, 'max', 'average'],
        value=None,
        description='Pooling',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    target_index_widget = widgets.IntSlider(
        min=0,
        max=len(dataset) - 1,
        step=1,
        value=initial_target_index,
        description='Image Index',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    # Arrange widgets in two columns
    left_column = VBox([layer_widget, target_index_widget], layout=widgets.Layout(width=fixed_width))
    right_column = VBox([activation_widget, pooling_widget], layout=widgets.Layout(width=fixed_width))

    # Combine columns into one horizontal box with fixed width
    ui = HBox([left_column, right_column], layout=widgets.Layout(width=fixed_width))

    # Combine UI and visualization into a single vertical box with fixed width
    ui_and_visualization = VBox([ui], layout=widgets.Layout(width=fixed_width))

    # Display the arranged widgets after setting up the layout
    display(ui_and_visualization)

    # Attach update function to widget value changes
    layer_widget.observe(update_visualization, names='value')
    activation_widget.observe(update_visualization, names='value')
    pooling_widget.observe(update_visualization, names='value')
    target_index_widget.observe(update_visualization, names='value')

    # Initial visualization
    update_visualization()


'''
def plot_training_metrics(df, y_vars, x_vars='epoch', figsize=(5, 4), smooth=0):
    """
    Plot training metrics from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing training results.
    y_vars (list or list of list of str): Columns to plot, e.g. ['train loss', 'test loss'] or [['train loss', 'test loss'], ['train ACC', 'test ACC']].
    x_vars (str or list of str): Column(s) to use for the x-axis. Defaults to 'epoch'. If a single string is provided, it will be used for all plots if necessary.
    figsize (tuple): Size of each subplot. Defaults to (5, 4).
    smooth (float): Smoothing parameter between 0 and 1 for exponential smoothing. Defaults to 0 (no smoothing).
    """
    # Validate input
    if isinstance(x_vars, str):
        x_vars = [x_vars] * len(y_vars)
    
    if len(x_vars) != len(y_vars):
        raise ValueError("The length of x_vars must match the length of y_vars.")
    
    for x_var in x_vars:
        if x_var not in df.columns:
            raise ValueError(f"The x-axis variable '{x_var}' is not in the DataFrame.")
    
    if not isinstance(y_vars[0], list):
        y_vars = [y_vars]  # Wrap in a list to treat as a single plot
    
    for y_list in y_vars:
        for y in y_list:
            if y not in df.columns:
                raise ValueError(f"The y-axis variable '{y}' is not in the DataFrame.")
    
    if len(y_vars) > 2:
        raise ValueError("Only up to two sets of y-axis variables can be plotted side by side.")
    
    # Plotting
    fig, axes = plt.subplots(1, len(y_vars), figsize=(figsize[0] * len(y_vars), figsize[1]))
    if len(y_vars) == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    linestyles = ['-', '--', '-.', ':']
    
    for ax, x_var, y_list in zip(axes, x_vars, y_vars):
        for idx, y in enumerate(y_list):
            if smooth > 0:
                alpha=(1-smooth/10)
                smoothed_values = df[y].ewm(alpha=alpha).mean()
                sns.lineplot(data=df, x=x_var, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=y)
            else:
                sns.lineplot(data=df, x=x_var, y=y, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=y)
        ax.set_xlabel(x_var)
        ax.set_ylabel('Metric')
        ax.set_title(', '.join(y_list))
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_metrics(dfs, y_vars, x_vars='epoch', figsize=(5, 4), smooth=0, df_labels=None, sns_theme='whitegrid'):
    """
    Plot training metrics from multiple DataFrames on the same axes.

    Creates publication-ready plots of training metrics (loss, accuracy, etc.) over epochs.
    Essential for visualizing training progress and diagnosing overfitting/underfitting.
    Used in every lesson to analyze model performance and compare experiments.

    Args:
        dfs (pd.DataFrame or list): Single DataFrame or list of DataFrames with training results.
        y_vars (list): Metric names to plot, or list of lists for subplots.
        x_vars (str): Column name for x-axis. Defaults to 'epoch'.
        figsize (tuple): Figure size (width, height). Defaults to (5, 4).
        smooth (float): Smoothing factor 0-1 for exponential smoothing. Defaults to 0.
        df_labels (list): Labels for multiple DataFrames. Defaults to None.
        sns_theme (str): Seaborn theme. Defaults to 'whitegrid'.

    Returns:
        None: Displays the plot using matplotlib.

    Example:
        Basic training visualization (L01_2_Nonlinear_Regression_1D):

        ```python
        from introdl.visul import plot_training_metrics

        # Load and plot training results
        results_df = load_results(checkpoint_file)
        plot_training_metrics(results_df, [['train loss', 'test loss']])

        # Plot multiple metrics in subplots
        plot_training_metrics(results_df,
                            [['train loss', 'test loss'],
                             ['train MAE', 'test MAE']])
        ```

        CNN training analysis (L02_3_MNIST_CNN):
        ```python
        # Visualize CNN training progress
        results_df = load_results(MODELS_PATH / 'L02_MNIST_CNN.pt')
        plot_training_metrics(results_df,
                            [['train loss', 'test loss'],
                             ['train accuracy', 'test accuracy']],
                            figsize=(12, 4))
        ```

        Compare experiments (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Compare SGD vs Adam performance
        sgd_results = load_results(MODELS_PATH / 'cifar10_sgd.pt')
        adam_results = load_results(MODELS_PATH / 'cifar10_adam.pt')

        plot_training_metrics([sgd_results, adam_results],
                            [['test loss']],
                            df_labels=['SGD', 'Adam'],
                            figsize=(8, 5))
        ```
    """
    # Set the seaborn theme
    sns.set_theme(style=sns_theme)
    sns.set_palette('colorblind')
    
    # Ensure dfs is a list of DataFrames
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    
    # Validate input
    if isinstance(x_vars, str):
        x_vars = [x_vars] * len(y_vars)
    
    if len(x_vars) != len(y_vars):
        raise ValueError("The length of x_vars must match the length of y_vars.")
    
    if not isinstance(y_vars[0], list):
        y_vars = [y_vars]  # Wrap in a list to treat as a single plot
    
    if len(y_vars) > 2:
        raise ValueError("Only up to two sets of y-axis variables can be plotted side by side.")
    
    if df_labels is None:
        df_labels = [f'DF {i + 1}' for i in range(len(dfs))]
    elif len(df_labels) != len(dfs):
        raise ValueError("The length of df_labels must match the length of dfs.")
    
    for df in dfs:
        for x_var in x_vars:
            if x_var not in df.columns:
                raise ValueError(f"The x-axis variable '{x_var}' is not in one of the DataFrames.")
        for y_list in y_vars:
            for y in y_list:
                if y not in df.columns:
                    raise ValueError(f"The y-axis variable '{y}' is not in one of the DataFrames.")
    
    # Plotting
    fig, axes = plt.subplots(1, len(y_vars), figsize=(figsize[0] * len(y_vars), figsize[1]))
    if len(y_vars) == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    linestyles = ['-', '--', '-.', ':']
    if len(dfs) == 1:
        colors = None  # Let seaborn handle different hues and styles for a single DataFrame
    else:
        colors = sns.color_palette(n_colors=len(dfs))
    
    for ax, x_var, y_list in zip(axes, x_vars, y_vars):
        for df_idx, df in enumerate(dfs):
            for idx, y in enumerate(y_list):
                if smooth > 0:
                    smoothed_values = df[y].ewm(alpha=smooth).mean()
                    label = y if len(dfs) == 1 else (f'{df_labels[df_idx]}' if len(y_list) == 1 else f'{y} - {df_labels[df_idx]}')
                    sns.lineplot(data=df, x=x_var, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], color=colors[df_idx] if colors else None, ax=ax, label=label)
                else:
                    label = y if len(dfs) == 1 else (f'{df_labels[df_idx]}' if len(y_list) == 1 else f'{y} - {df_labels[df_idx]}')
                    sns.lineplot(data=df, x=x_var, y=y, linestyle=linestyles[idx % len(linestyles)], color=colors[df_idx] if colors else None, ax=ax, label=label)
        ax.set_xlabel(x_var)
        ax.set_ylabel('Metric')
        ax.set_title(', '.join(y_list))
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# === REMOVED_FUNCTION_START: plot_training_metrics_widget ===
# def plot_training_metrics_widget(df=None, numplots=2, figsize=(5, 4)):
#     """
#     Visualize training results using an interactive widget with one or two plots, including save functionality.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame containing training results.
#     numplots (int): Number of plots to display (1 or 2). Defaults to 2.
#     figsize (tuple): Size of each subplot. Defaults to (5, 4).
#     """
#     if numplots not in [1, 2]:
#         raise ValueError("numplots can only be 1 or 2.")
    
#     if df is None:
#         print("Please provide a DataFrame.")
#         return
    
#     # Output widget to display the plots
#     output_plot = Output(layout=Layout(height='400px', width='100%'))
#     output_messages = Output(layout=Layout(height='50px'))  # Messages area

#     # Function to initialize the widget with the provided DataFrame
#     def _initialize_widget(df):
#         clear_output(wait=True)  # Clear previous outputs
        
#         col_width = '200px'

#         # Dropdowns for plot configuration
#         x_axis_dropdown_1 = Dropdown(
#             options=['epoch', 'total time'] if 'epoch' in df.columns else df.columns,
#             value='epoch' if 'epoch' in df.columns else df.columns[0],
#             description='X-axis (Plot 1):',
#             layout=Layout(width=col_width)
#         )
        
#         y_axis_select_1 = SelectMultiple(
#             options=df.columns,
#             value=['train loss'] if 'train loss' in df.columns else [df.columns[0]],
#             description='Y-axis (Plot 1):',
#             layout=Layout(width=col_width, height='150px')
#         )
        
#         x_axis_dropdown_2 = None
#         y_axis_select_2 = None
#         save_options = None
#         if numplots == 2:
#             x_axis_dropdown_2 = Dropdown(
#                 options=['epoch', 'total time'] if 'epoch' in df.columns else df.columns,
#                 value='epoch' if 'epoch' in df.columns else df.columns[0],
#                 description='X-axis (Plot 2):',
#                 layout=Layout(width=col_width)
#             )
#             y_axis_select_2 = SelectMultiple(
#                 options=df.columns,
#                 value=['train loss'] if 'train loss' in df.columns else [df.columns[0]],
#                 description='Y-axis (Plot 2):',
#                 layout=Layout(width=col_width, height='150px')
#             )
#             # Radio buttons for selecting which plot(s) to save
#             save_options = RadioButtons(
#                 options=['Left Plot', 'Right Plot', 'Both'],
#                 description='Save:',
#                 disabled=False,
#                 layout=Layout(width=col_width)
#             )
        
#         # Text box to enter filename
#         filename_input = Text(description="Filename:", value="training_results.png", layout=Layout(width='300px'))
        
#         # Save button
#         save_button = Button(description="Save Figure", button_style="success", layout=Layout(width=col_width))

#         # Smoothing slider
#         smoothing_slider = IntSlider(
#             value=0,
#             min=0,
#             max=9,  # Max smoothing value
#             step=1,  # Step size
#             description='Smoothing:',
#             layout=Layout(width='300px')
#         )

#         # Function to save the figure with the selected options
#         def _save_figure(_):
#             with output_messages:
#                 try:
#                     linestyles = ['-', '--', '-.', ':']
#                     filename = filename_input.value
#                     if not filename.endswith(".png"):
#                         filename += ".png"
                    
#                     alpha = (1-smoothing_slider.value/10)

#                     if numplots == 1:
#                         fig, ax = plt.subplots(figsize=figsize)
#                         for idx, col in enumerate(y_axis_select_1.value):
#                             smoothed_values = df[col].ewm(alpha=alpha).mean() if smoothing_slider.value > 0 else df[col]
#                             sns.lineplot(data=df, x=x_axis_dropdown_1.value, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
#                         ax.set_xlabel(x_axis_dropdown_1.value)
#                         ax.set_ylabel('Metric')
#                         ax.set_title('Plot')
#                         ax.legend()
#                         ax.grid(True)
#                         plt.tight_layout()
#                         fig.savefig(filename, bbox_inches='tight')
#                         print(f"Plot saved as {filename}")
#                         plt.close(fig)
#                     elif numplots == 2:
#                         save_choice = save_options.value
#                         if save_choice == 'Left Plot':
#                             fig, ax = plt.subplots(figsize=figsize)
#                             for idx, col in enumerate(y_axis_select_1.value):
#                                 smoothed_values = df[col].ewm(alpha=alpha).mean() if smoothing_slider.value > 0 else df[col]
#                                 sns.lineplot(data=df, x=x_axis_dropdown_1.value, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
#                             ax.set_xlabel(x_axis_dropdown_1.value)
#                             ax.set_ylabel('Metric')
#                             ax.set_title('Left Plot')
#                             ax.legend()
#                             ax.grid(True)
#                             plt.tight_layout()
#                             fig.savefig(filename, bbox_inches='tight')
#                             print(f"Left plot saved as {filename}")
#                             plt.close(fig)
#                         elif save_choice == 'Right Plot':
#                             fig, ax = plt.subplots(figsize=figsize)
#                             for idx, col in enumerate(y_axis_select_2.value):
#                                 smoothed_values = df[col].ewm(alpha=alpha).mean() if smoothing_slider.value > 0 else df[col]
#                                 sns.lineplot(data=df, x=x_axis_dropdown_2.value, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
#                             ax.set_xlabel(x_axis_dropdown_2.value)
#                             ax.set_ylabel('Metric')
#                             ax.set_title('Right Plot')
#                             ax.legend()
#                             ax.grid(True)
#                             plt.tight_layout()
#                             fig.savefig(filename, bbox_inches='tight')
#                             print(f"Right plot saved as {filename}")
#                             plt.close(fig)
#                         else:  # Both plots
#                             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
#                             for idx, col in enumerate(y_axis_select_1.value):
#                                 smoothed_values = df[col].ewm(alpha=alpha).mean() if smoothing_slider.value > 0 else df[col]
#                                 sns.lineplot(data=df, x=x_axis_dropdown_1.value, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax1, label=col)
#                             ax1.set_xlabel(x_axis_dropdown_1.value)
#                             ax1.set_ylabel('Metric')
#                             ax1.set_title('Left Plot')
#                             ax1.legend()
#                             ax1.grid(True)

#                             for idx, col in enumerate(y_axis_select_2.value):
#                                 smoothed_values = df[col].ewm(alpha=alpha).mean() if smoothing_slider.value > 0 else df[col]
#                                 sns.lineplot(data=df, x=x_axis_dropdown_2.value, y=smoothed_values, linestyle=linestyles[idx % len(linestyles)], ax=ax2, label=col)
#                             ax2.set_xlabel(x_axis_dropdown_2.value)
#                             ax2.set_ylabel('Metric')
#                             ax2.set_title('Right Plot')
#                             ax2.legend()
#                             ax2.grid(True)
                            
#                             plt.tight_layout()
#                             fig.savefig(filename, bbox_inches='tight')
#                             print(f"Both plots saved as {filename}")
#                             plt.close(fig)
#                 except Exception as e:
#                     print(f"Error saving figure: {e}")
        
#         # Link the save button to the save_figure function
#         save_button.on_click(_save_figure)

#         # Function to plot data based on selected x-axis and y-axis using the plot_training_metrics function
#         def _plot_data(_=None):
#             with output_plot:
#                 clear_output(wait=True)
#                 y_vars = [y for y in y_axis_select_1.value]
#                 x_vars = [x_axis_dropdown_1.value]
#                 if numplots == 2:
#                     y_vars = [y_vars, [y for y in y_axis_select_2.value]]
#                     x_vars.append(x_axis_dropdown_2.value)
#                 else:
#                     y_vars = [y_vars]  # Wrap in a list to treat as a single plot
#                 plot_training_metrics(df, y_vars, x_vars=x_vars, figsize=figsize, smooth=smoothing_slider.value)
        
#         # Link dropdowns and slider to the plot function
#         x_axis_dropdown_1.observe(_plot_data, names='value')
#         y_axis_select_1.observe(_plot_data, names='value')
#         smoothing_slider.observe(_plot_data, names='value')
#         if numplots == 2:
#             x_axis_dropdown_2.observe(_plot_data, names='value')
#             y_axis_select_2.observe(_plot_data, names='value')

#         # Layout for dropdowns and buttons
#         if numplots == 2:
#             plot_controls = HBox([VBox([x_axis_dropdown_1, y_axis_select_1]), VBox([x_axis_dropdown_2, y_axis_select_2]), VBox([save_options, filename_input, save_button]), VBox([smoothing_slider])])
#         else:
#             plot_controls = HBox([VBox([x_axis_dropdown_1, y_axis_select_1]), VBox([filename_input, save_button]), VBox([smoothing_slider])])

#         # Display the layout: top for plots, bottom for controls
#         display(VBox([output_plot, plot_controls, output_messages]))
        
#         # Initial plot display
#         _plot_data()
    
#     # Initialize the widget
#     _initialize_widget(df)
# === REMOVED_FUNCTION_END: plot_training_metrics_widget ===


#################################################
# Interactive widget for MNIST digit generation
#################################################

'''
# Function to extract 28x28 array from the canvas and binarize it
def canvas_to_binarized_array(canvas, grid_size=28):
    data = np.array(canvas.get_image_data())
    alpha_channel = data[:, :, 3]
    
    # Calculate the current cell size dynamically based on the canvas size
    canvas_width, canvas_height = canvas.width, canvas.height
    cell_size_x = canvas_width // grid_size
    cell_size_y = canvas_height // grid_size

    # Ensure reshaping works with the current cell sizes
    downsampled = alpha_channel.reshape((grid_size, cell_size_y, grid_size, cell_size_x)).mean(axis=(1, 3))
    
    # Binarize the array: Set to 1 if the average value is above a threshold (e.g., 128), else 0
    binarized_array = (downsampled > 128).astype(np.float32)
    
    return binarized_array

# Function to convert the binarized array to a PyTorch tensor for the model
def binarized_array_to_tensor(binarized_array):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(binarized_array).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to predict the digit and softmax probabilities using the trained PyTorch model
def predict_digit(model, canvas, output_widget, plot_widget):
    binarized_array = canvas_to_binarized_array(canvas)
    image_tensor = binarized_array_to_tensor(binarized_array)
    
    # Get the model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        softmax_probs = F.softmax(output, dim=1).numpy().flatten()  # Get softmax probabilities
    
    predicted_digit = output.argmax(dim=1).item()
    
    # Display the predicted digit
    output_widget.clear_output()
    with output_widget:
        print(f'Predicted Digit: {predicted_digit}')
    
    # Update the softmax probabilities plot
    plot_softmax_probabilities(softmax_probs, plot_widget)

# Function to plot the softmax probabilities as a bar chart
def plot_softmax_probabilities(probs, plot_widget):
    with plot_widget:
        plot_widget.clear_output(wait=True)  # Clear previous plot
        fig, ax = plt.subplots(figsize=(3, 2))  # Make the plot a bit smaller
        ax.bar(range(10), probs, color='blue')
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_title("Softmax Probabilities")
        ax.set_ylim([0, 1])  # Set y-axis range to [0, 1]
        ax.set_xlim([-0.5, 9.5])  # Ensure the x-axis stays within 0-9
        ax.grid(True, axis='y')  # Add grid for better visibility
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Close the figure after displaying it

# Function to reset the bar graph to default probabilities (0.1 for each digit)
def reset_softmax_probabilities(plot_widget):
    default_probs = np.full(10, 0.1)  # Set default probabilities to 0.1
    plot_softmax_probabilities(default_probs, plot_widget)

# Main function to set up the drawing and prediction interface
def interactive_mnist_prediction(model):
    instructions = widgets.HTML(value="<h3>Draw digit.</h3>")
    
    square_size = 196  # Set a fixed square size for the canvas
    grid_size = 28  # Keep the grid size as 28x28
    cell_size = square_size // grid_size  # Dynamically calculate cell size
    
    # Set canvas to have the same width and height to ensure it's always square
    canvas = Canvas(width=square_size, height=square_size, background_color='black', sync_image_data=True)
    
    # Explicitly set the layout to maintain a square shape
    canvas.layout.width = f'{square_size}px'
    canvas.layout.height = f'{square_size}px'
    canvas.layout.border = '3px solid blue'  # Blue border to frame the drawing area

    # Draw the grid on the full square canvas area
    def draw_grid():
        canvas.stroke_style = 'blue'
        canvas.stroke_rect(0, 0, square_size, square_size)
        canvas.stroke_style = 'black'
        for x in range(0, square_size, cell_size):
            canvas.stroke_line(x, 0, x, square_size)
        for y in range(0, square_size, cell_size):
            canvas.stroke_line(0, y, square_size, y)

    draw_grid()

    # Radio buttons to select pen width
    pen_width_selector = widgets.RadioButtons(
        options=[('1', 1), ('2', 2)],
        description='Pen Width:',
        disabled=False
    )

    clear_button = widgets.Button(description='Clear')
    predict_button = widgets.Button(description='Predict')
    output_widget = widgets.Output()
    plot_widget = Output()  # For displaying the softmax probabilities plot

    # Initialize the softmax probability plot with default values (0.1 for each digit)
    reset_softmax_probabilities(plot_widget)

    control_box = VBox([instructions, pen_width_selector, clear_button, predict_button, output_widget])
    hbox_layout = HBox([canvas, control_box, plot_widget])

    # Clear the canvas when the clear button is clicked
    def clear_canvas(b):
        canvas.clear()
        draw_grid()
        reset_softmax_probabilities(plot_widget)  # Reset probabilities to default (0.1 for each digit)
    
    clear_button.on_click(clear_canvas)

    # Predict the digit when the predict button is clicked
    def predict_digit_button(b):
        predict_digit(model, canvas, output_widget, plot_widget)
    
    predict_button.on_click(predict_digit_button)

    # Fill in multiple grid cells on mouse down or move, based on selected pen width
    drawing = False
    
    def on_mouse_down(x, y):
        nonlocal drawing
        drawing = True
        fill_grid_cells(x, y)

    def on_mouse_up(x, y):
        nonlocal drawing
        drawing = False

    def on_mouse_move(x, y):
        if drawing:
            fill_grid_cells(x, y)

    def fill_grid_cells(x, y):
        """Fill a square of size `pen_size x pen_size` grid cells around (x, y) based on selected pen width"""
        pen_size = pen_width_selector.value  # Get the selected pen width from the radio buttons
        if 0 <= x < square_size and 0 <= y < square_size:
            grid_x = int(x // cell_size)
            grid_y = int(y // cell_size)
            
            for i in range(-pen_size//2 + 1, pen_size//2 + 1):
                for j in range(-pen_size//2 + 1, pen_size//2 + 1):
                    nx, ny = grid_x + i, grid_y + j
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        start_x = nx * cell_size
                        start_y = ny * cell_size
                        canvas.fill_rect(start_x, start_y, cell_size, cell_size)

    # Bind the event handlers
    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_up(on_mouse_up)
    canvas.on_mouse_move(on_mouse_move)

    display(hbox_layout)
'''

# Function to extract 28x28 array from the canvas and binarize it
def canvas_to_binarized_array(canvas, grid_size=28):
    data = np.array(canvas.get_image_data())
    alpha_channel = data[:, :, 3]
    
    # Calculate the current cell size dynamically based on the canvas size
    canvas_width, canvas_height = canvas.width, canvas.height
    cell_size_x = canvas_width // grid_size
    cell_size_y = canvas_height // grid_size

    # Ensure reshaping works with the current cell sizes
    downsampled = alpha_channel.reshape((grid_size, cell_size_y, grid_size, cell_size_x)).mean(axis=(1, 3))
    
    # Binarize the array: Set to 1 if the average value is above a threshold (e.g., 128), else 0
    binarized_array = (downsampled > 128).astype(np.float32)
    
    return binarized_array

# Function to convert the binarized array to a PyTorch tensor for the model
def binarized_array_to_tensor(binarized_array):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(binarized_array).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to predict the digit and softmax probabilities using the trained PyTorch model
def predict_digit(model, canvas, output_widget, plot_widget):
    binarized_array = canvas_to_binarized_array(canvas)
    image_tensor = binarized_array_to_tensor(binarized_array)
    
    # Get the model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        softmax_probs = F.softmax(output, dim=1).numpy().flatten()  # Get softmax probabilities
    
    predicted_digit = output.argmax(dim=1).item()
    
    # Display the predicted digit
    output_widget.clear_output()
    with output_widget:
        print(f'Predicted Digit: {predicted_digit}')
    
    # Update the softmax probabilities plot
    plot_softmax_probabilities(softmax_probs, plot_widget)

# Function to plot the softmax probabilities as a bar chart
def plot_softmax_probabilities(probs, plot_widget):
    with plot_widget:
        plot_widget.clear_output(wait=True)  # Clear previous plot
        fig, ax = plt.subplots(figsize=(3, 2))  # Make the plot a bit smaller
        ax.bar(range(10), probs, color='blue')
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_title("Softmax Probabilities")
        ax.set_ylim([0, 1])  # Set y-axis range to [0, 1]
        ax.set_xlim([-0.5, 9.5])  # Ensure the x-axis stays within 0-9
        ax.grid(True, axis='y')  # Add grid for better visibility
        plt.tight_layout()
        plt.show()

# Function to reset the bar graph to default probabilities (0.1 for each digit)
def reset_softmax_probabilities(plot_widget):
    default_probs = np.full(10, 0.1)  # Set default probabilities to 0.1
    plot_softmax_probabilities(default_probs, plot_widget)

# Main function to set up the drawing and prediction interface
def interactive_mnist_prediction(model):
    instructions = widgets.HTML(value="<h3>Draw digit.</h3>")
    
    square_size = 196  # Set a fixed square size for the canvas
    grid_size = 28  # Keep the grid size as 28x28
    cell_size = square_size // grid_size  # Dynamically calculate cell size
    
    # Set canvas to have the same width and height to ensure it's always square
    canvas = Canvas(width=square_size, height=square_size, background_color='black', sync_image_data=True)
    
    # Explicitly set the layout to maintain a square shape
    canvas.layout.width = f'{square_size}px'
    canvas.layout.height = f'{square_size}px'
    canvas.layout.border = '3px solid blue'  # Blue border to frame the drawing area

    # Draw the grid on the full square canvas area
    def draw_grid():
        canvas.stroke_style = 'blue'
        canvas.stroke_rect(0, 0, square_size, square_size)
        canvas.stroke_style = 'lightgray'
        for x in range(0, square_size, cell_size):
            canvas.stroke_line(x, 0, x, square_size)
        for y in range(0, square_size, cell_size):
            canvas.stroke_line(0, y, square_size, y)

    draw_grid()

    # Radio buttons to select pen width
    pen_width_selector = widgets.RadioButtons(
        options=[('1', 1), ('2', 2)],
        description='Pen Width:',
        disabled=False
    )

    clear_button = widgets.Button(description='Clear')
    predict_button = widgets.Button(description='Predict')
    output_widget = widgets.Output()
    plot_widget = Output()  # For displaying the softmax probabilities plot

    # Initialize the softmax probability plot with default values (0.1 for each digit)
    reset_softmax_probabilities(plot_widget)

    control_box = VBox([instructions, pen_width_selector, clear_button, predict_button, output_widget])
    hbox_layout = HBox([canvas, control_box, plot_widget])

    # Clear the canvas when the clear button is clicked
    def clear_canvas(b):
        canvas.clear()
        draw_grid()
        reset_softmax_probabilities(plot_widget)  # Reset probabilities to default (0.1 for each digit)
    
    clear_button.on_click(clear_canvas)

    # Predict the digit when the predict button is clicked
    def predict_digit_button(b):
        predict_digit(model, canvas, output_widget, plot_widget)
    
    predict_button.on_click(predict_digit_button)

    # Fill in multiple grid cells on mouse down or move, based on selected pen width
    drawing = False
    
    def on_mouse_down(x, y):
        nonlocal drawing
        drawing = True
        fill_grid_cells(x, y)

    def on_mouse_up(x, y):
        nonlocal drawing
        drawing = False

    def on_mouse_move(x, y):
        if drawing:
            fill_grid_cells(x, y)

    def fill_grid_cells(x, y):
        """Fill a square of size `pen_size x pen_size` grid cells around (x, y) based on selected pen width"""
        pen_size = pen_width_selector.value  # Get the selected pen width from the radio buttons
        if 0 <= x < square_size and 0 <= y < square_size:
            grid_x = int(x // cell_size)
            grid_y = int(y // cell_size)
            
            for i in range(-pen_size//2 + 1, pen_size//2 + 1):
                for j in range(-pen_size//2 + 1, pen_size//2 + 1):
                    nx, ny = grid_x + i, grid_y + j
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        start_x = nx * cell_size
                        start_y = ny * cell_size
                        canvas.fill_rect(start_x, start_y, cell_size, cell_size)

    # Bind the event handlers
    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_up(on_mouse_up)
    canvas.on_mouse_move(on_mouse_move)

    display(hbox_layout)

