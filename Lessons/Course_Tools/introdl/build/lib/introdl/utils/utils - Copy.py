import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
import sys
import os
import random
import numpy as np
import pandas as pd
import inspect
from torchinfo import summary
import traceback
from textwrap import TextWrapper
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
from huggingface_hub import hf_hub_download
import warnings
import shutil

try:
    import dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
finally:
    from dotenv import load_dotenv

###########################################################
# Utility Functions
###########################################################

def detect_jupyter_environment():
    if 'google.colab' in sys.modules:
        return "colab"
    elif 'VSCODE_PID' in os.environ:
        return "vscode"
    elif 'COCALC_CODE_PORT' in os.environ:
        return "cocalc"
    elif 'JPY_PARENT_PID' in os.environ:
        return "jupyterlab"
    else:
        return "Unknown"

def config_paths_keys(env_path="~/Lessons/Course_Tools/local.env", api_keys_env="~/Lessons/Course_Tools/api_keys.env"):
    """
    Reads environment variables and sets paths.
    If variables are not set, it uses dotenv to load them based on the environment:
    - CoCalc: ~/Lessons/Course_Tools/cocalc.env
    - Colab: ~/Lessons/Course_Tools/colab.env
    - Other: ~/Lessons/Course_Tools/local.env (default)

    Additionally, loads API keys from api_keys_env if HF_TOKEN and OPENAI_API_KEY are not already set.

    Parameters:
        env_path (str): Path to the local environment file, defaulting to ~/Lessons/Course_Tools/local.env.
        api_keys_env (str): Path to the API keys environment file, defaulting to ~/Lessons/Course_Tools/api_keys.env.

    Returns:
        dict: A dictionary with keys 'MODELS_PATH' and 'DATA_PATH'.
    """
    # Determine the environment
    environment = detect_jupyter_environment()
    if environment == "cocalc":
        env_file = Path("~/Lessons/Course_Tools/cocalc.env").expanduser()
    elif environment == "colab":
        env_file = Path("~/Lessons/Course_Tools/colab.env").expanduser()
    else:
        env_file = Path(env_path).expanduser()

    # Load the environment variables from the determined .env file
    load_dotenv(env_file, override=False)

    # Load API keys if not already set
    if not os.getenv('HF_TOKEN') or not os.getenv('OPENAI_API_KEY'):
        load_dotenv(api_keys_env, override=False)

    # Retrieve and expand paths
    models_path = Path(os.getenv('MODELS_PATH', "")).expanduser()
    data_path = Path(os.getenv('DATA_PATH', "")).expanduser()
    torch_home = Path(os.getenv('TORCH_HOME', "")).expanduser()
    hf_home = Path(os.getenv('HF_HOME', "")).expanduser()

    # Set environment variables to expanded paths
    os.environ['MODELS_PATH'] = str(models_path)
    os.environ['DATA_PATH'] = str(data_path)
    os.environ['TORCH_HOME'] = str(torch_home)
    os.environ['HF_HOME'] = str(hf_home)

    # Create directories if they don't exist
    for path in [models_path, data_path, torch_home, hf_home]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    # Ensure paths are set
    print(f"MODELS_PATH={models_path}")
    print(f"DATA_PATH={data_path}")
    print(f"TORCH_HOME={torch_home}")
    print(f"HF_HOME={hf_home}")

    return {
        'MODELS_PATH': models_path,
        'DATA_PATH': data_path
    }
 
def get_device():
    """
    Returns the appropriate device ('cuda', 'mps', or 'cpu') depending on availability.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_results(checkpoint_file, device=torch.device('cpu'), repo_id=None, filename=None):
    """
    Load the results from a checkpoint file.

    Parameters:
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - filename (str, optional): Filename in the Hugging Face repository if different from checkpoint_file.

    Returns:
    - results (pd.DataFrame): The loaded results from the checkpoint file.
    """
    import os

    # Attempt to download from Hugging Face Hub if not available locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally and no repo_id provided.")
        # Attempt to download from Hugging Face Hub
        if not filename:
            filename = os.path.basename(checkpoint_file)
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, token=os.getenv('HF_TOKEN'))

        # Move downloaded file to the original checkpoint_file location
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(downloaded_file, checkpoint_file)
        
    # Suppress FutureWarning during torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # Extract the results
    if 'results' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'results'.")
    return pd.DataFrame(checkpoint_dict['results'])

'''
def load_results(checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", filename=None):
    """
    Load the results from a checkpoint file.

    Parameters:
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - filename (str, optional): Filename in the Hugging Face repository if different from checkpoint_file.

    Returns:
    - results (pd.DataFrame): The loaded results from the checkpoint file.
    """
    import os

    # Attempt to download from Hugging Face Hub if not available locally
    if not os.path.exists(checkpoint_file) and repo_id:
        if not filename:
            filename = os.path.basename(checkpoint_file)
        checkpoint_file = hf_hub_download(repo_id=repo_id, filename=filename)

    # Allowlist numpy scalar
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

    # Load the checkpoint with weights_only=True
    checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)

    # Extract the results from the checkpoint
    if 'results' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'results'.")
    return pd.DataFrame(checkpoint_dict['results'])
'''

'''
def load_results(ckpt_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", filename=None):
    """
    Load the results from a checkpoint file. Downloads the file from Hugging Face Hub if it does not exist locally.
    
    Parameters:
    - ckpt_file (str): The local path where the file would typically be saved (for reference).
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face model repository identifier.
    - filename (str, optional): The filename in the Hugging Face repository (if different from ckpt_file name).
    
    Returns:
    - pd.DataFrame: The loaded results as a DataFrame.
    """
    if filename is None:
        filename = ckpt_file.name if hasattr(ckpt_file, 'name') else os.path.basename(ckpt_file)
    
    # Download the file or retrieve its cached path
    hf_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    #print(f"Using file from Hugging Face Hub cache: {hf_file_path}")
    
    # Load the checkpoint directly from the cached file
    checkpoint_dict = torch.load(hf_file_path, map_location=device)
    return pd.DataFrame(checkpoint_dict['results'])
'''

'''
def load_results(ckpt_file, device=torch.device('cpu')):
    """
    Load the results from a checkpoint file.

    Parameters:
    - ckpt_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.

    Returns:
    - results: The loaded results from the checkpoint file.
    """
    checkpoint_dict = torch.load(ckpt_file, map_location=device, weights_only=False) # moves tensors onto 'device'
    return pd.DataFrame(checkpoint_dict['results'])
'''

'''
def load_model(model, checkpoint_file, device=torch.device('cpu')):
    """
    Load the model from a checkpoint file.

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.

    Returns:
    - model: The loaded model from the checkpoint file.
    """
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")
        
    checkpoint_dict = torch.load(checkpoint_file, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint_dict['model_state_dict']) 
    return model.to(device)
'''

def load_model(model, checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", filename=None):
    """
    Load the model from a checkpoint file, trying locally first, then Hugging Face Hub if not found.

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - filename (str, optional): Filename in the Hugging Face repository if different from checkpoint_file.

    Returns:
    - model: The loaded model from the checkpoint file.
    """
    import os
    import inspect
    from huggingface_hub import hf_hub_download

    # Check if checkpoint file exists locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally and no repo_id provided.")
        # Attempt to download from Hugging Face Hub
        if not filename:
            filename = os.path.basename(checkpoint_file)
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, token=os.getenv('HF_TOKEN'))

        # Move downloaded file to the original checkpoint_file location
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(downloaded_file, checkpoint_file)

    # Instantiate model if a class is passed
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if 'model_state_dict' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    return model.to(device)

'''
def load_model(model, checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", filename=None):
    """
    Load the model from a checkpoint file, downloading it from Hugging Face Hub if not found locally.

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face model repository identifier.
    - filename (str, optional): The filename in the Hugging Face repository (if different from checkpoint_file name).

    Returns:
    - model: The loaded model from the checkpoint file.
    """
    if filename is None:
        filename = checkpoint_file.name if hasattr(checkpoint_file, 'name') else os.path.basename(checkpoint_file)
    
    # Download the file or retrieve its cached path
    hf_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    #print(f"Using file from Hugging Face Hub cache: {hf_file_path}")
    
    # Check if the input is a class or an instance of nn.Module
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")
        
    # Load the checkpoint and the model's state_dict
    checkpoint_dict = torch.load(hf_file_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    
    return model.to(device)
'''

'''
def summarizer(model, input_size, device=torch.device('cpu'), col_width=20):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    """
    model = model.to(device)
    print(summary(model, input_size=input_size, col_width=col_width, col_names=["input_size", "output_size", "num_params"]))
'''

def summarizer(model, input_size, device=torch.device('cpu'), col_width=20, verbose=False, varnames = True, **kwargs):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    - verbose (bool, optional): If True, display the full error stack trace; otherwise, show only a simplified error message. Defaults to False.
    - **kwargs: Additional keyword arguments to pass to the summary function.
    """
    model = model.to(device)
    try:
        colnames = ["input_size", "output_size", "num_params"]
        rowsettings = ["var_names"] if varnames else ["depth"]
        print(summary(model, input_size=input_size, col_width=col_width, row_settings=rowsettings, col_names=colnames, **kwargs))
    except RuntimeError as e:
        if verbose:
            # Print the full stack trace and original error message
            traceback.print_exc()
            print(f"Original Error: {e}")
        else:
            # Display simplified error message with additional message for verbose option
            error_message = str(e).splitlines()[-1].replace("See above stack traces for more details.", "").strip()
            error_message = error_message.replace("Failed to run torchinfo.", "Failed to run all model layers.")
            error_message += " Run again with verbose=True to see stack trace."
            print(f"Error: {error_message}")

def classifier_predict(dataset, model, device, batch_size=32, return_labels=False):
    """
    Collects predictions from a PyTorch dataset using a classification model.
    Optionally returns ground truth labels.

    Assumptions:
        - The model outputs logits for each class (not probabilities or class indices).
        - The dataset returns tuples of (inputs, labels) where labels are integers representing class indices.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to evaluate.
        model (torch.nn.Module): The classification model. Assumes outputs are logits for each class.
        device (torch.device): The device to run the evaluation on.
        return_labels (bool): Whether to return ground truth labels along with predictions.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: Predicted labels (class indices).
        list (optional): Ground truth labels (if return_labels=True).
    """
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize lists to store predictions and ground truth labels
    predictions = []
    ground_truth = [] if return_labels else None

    # Turn off gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            if return_labels:
                labels = labels.to(device)

            # Forward pass through the model
            logits = model(inputs)

            # Get predicted labels (the class with the highest logit)
            preds = torch.argmax(logits, dim=1)

            # Append predictions to the list
            predictions.extend(preds.cpu().tolist())
            # Append ground truth labels if requested
            if return_labels:
                ground_truth.extend(labels.cpu().tolist())

    if return_labels:
        return predictions, ground_truth
    return predictions

def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, persistent_workers = True, 
                           use_augmentation=False):
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.v2.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.v2.Compose, optional): Transformations for the validation set. Defaults to None.
        valid_prop (float or None): Proportion of the training set to use for validation. If 0.0 or None, no validation split is made.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to download/load CIFAR10 data.
        downsample_prop (float): Proportion of the dataset to keep if less than 1. Defaults to 1.0.
        num_workers (int): Number of worker processes to use for data loading.
        use_augmentation (bool): Whether to apply data augmentation to the training set. Defaults to False.

    Returns:
        tuple: Train loader, test loader, and optionally valid loader, along with the datasets.
    """

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.2010)

    if transform_train is None:
        if use_augmentation:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=mean, std=std), 
                transforms.ToPureTensor()   
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
                transforms.ToPureTensor()   
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPureTensor()   
        ])
        
    # Set validation transform; if None, use transform_test
    if transform_valid is None:
        transform_valid = transform_test

    # Load the full training and test datasets
    train_dataset_full = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Generate indices for training and validation if needed
    train_indices, valid_indices = None, None
    if valid_prop and 0 < valid_prop < 1.0:
        total_indices = list(range(len(train_dataset_full)))
        train_indices, valid_indices = train_test_split(
            total_indices,
            test_size=valid_prop,
            random_state=seed,
            shuffle=True
        )

    # Downsample datasets if required
    if downsample_prop < 1.0:
        train_indices = train_indices[:int(downsample_prop * len(train_indices))] if train_indices else None
        valid_indices = valid_indices[:int(downsample_prop * len(valid_indices))] if valid_indices else None

    # Create Subset datasets for training and optionally validation
    train_dataset = Subset(train_dataset_full, train_indices) if train_indices else train_dataset_full
    valid_dataset = Subset(CIFAR10(root=data_dir, train=True, download=True, transform=transform_valid), valid_indices) if valid_indices else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, persistent_workers=persistent_workers)
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, persistent_workers=persistent_workers)

    if valid_loader:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader

def wrap_print_text(print):
    """
    Wraps the given print function to format text with a specified width.
    This function takes a print function as an argument and returns a new function
    that formats the text to a specified width before printing. The text is wrapped
    to 80 characters per line, and long words are broken to fit within the width.
    Args:
        print (function): The original print function to be wrapped.
    Returns:
        function: A new function that formats text to 80 characters per line and
                  then prints it using the original print function.
    Example:
        wrapped_print = wrap_print_text(print)
        wrapped_print("This is a very long text that will be wrapped to fit within 80 characters per line.")
    Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

    def wrapped_func(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

    return wrapped_func