import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


# tdqm progress bars can behave differently in different environments so we change tdqm
# to a version that works in different environments

# Check if running in VSCode or JupyterLab
if 'VSCODE_PID' in os.environ:
    from tqdm import tqdm
else:
    from tqdm.autonotebook import tqdm

# Set Seaborn theme
sns.set_theme(style="darkgrid")

def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    plt.figure(figsize=(6,5))
    cs = plt.contourf(xv, yv, y_hat[:,1].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu_r)
    plt.colorbar(cs)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)

###########################################################
# Utility Functions (mostly for training)
###########################################################

def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None, use_tqdm=True):
    """
    Run one epoch of training or evaluation.

    Parameters:
        model (torch.nn.Module): The PyTorch model to run for one epoch.
        optimizer (torch.optim.Optimizer): The optimizer object that will update the weights of the network.
        data_loader (torch.utils.data.DataLoader): The DataLoader object that returns tuples of (input, label) pairs.
        loss_func (callable): The loss function that takes in two arguments, the model outputs and the labels, and returns a score.
        device (torch.device): The compute location to perform training.
        results (dict): A dictionary to store the results of the epoch.
        score_funcs (dict): A dictionary of scoring functions to use to evaluate the performance of the model.
        prefix (str, optional): A string to pre-fix to any scores placed into the results dictionary. Default is an empty string.
        desc (str, optional): A description to use for the progress bar. Default is None.
        use_tqdm (bool, optional): Whether to use tqdm for displaying the progress bar. Default is True.

    Returns:
        float: The time spent on the epoch.

    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    
    # Loop over batches
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False, disable=not use_tqdm):
        # Move the batch to the device we are using
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        # Forward pass
        y_hat = model(inputs)
        
        # Compute loss
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward() # Compute gradients
            optimizer.step() # Update the weights
            optimizer.zero_grad() # Zero the gradients

        # Store loss
        running_loss.append(loss.item())

        if score_funcs is not None and len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # Move labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            
            # Store true and predicted values for scoring
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    # End training epoch
    end = time.time()

    # Convert predictions to labels (if classification problem)
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:  # If classification, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    
    # Record loss and scores
    results[prefix + " loss"].append(np.mean(running_loss))
    if score_funcs is not None:
        for name, score_func in score_funcs.items():
            try:
                results[prefix + " " + name].append(score_func(y_true, y_pred))
            except:
                results[prefix + " " + name].append(float("NaN"))
    
    return end - start  # Return time spent on epoch


def train_simple_network(model, loss_func, train_loader, test_loader=None, score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, lr=0.001, use_tqdm=True):
    """
    Trains a simple neural network model using the specified loss function and data loaders.

    Args:
        model (torch.nn.Module): The neural network model to train.
        loss_func (torch.nn.Module): The loss function to optimize during training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader, optional): The data loader for the testing dataset. Defaults to None.
        score_funcs (list, optional): List of evaluation score functions to track during training. Defaults to None.
        epochs (int, optional): The number of training epochs. Defaults to 50.
        device (str, optional): The device to use for training (e.g., "cpu" or "cuda"). Defaults to "cpu".
        checkpoint_file (str, optional): The file path to save the model checkpoint. Defaults to None.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        use_tqdm (bool, optional): Whether to display a progress bar during training. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the training and testing results for each epoch.
    """
    
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")

    if score_funcs is not None:
        for eval_score in score_funcs:
            to_track.append("train " + eval_score )
            if test_loader is not None:
                to_track.append("test " + eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        
    #SGD is Stochastic Gradient Decent.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=not use_tqdm):
        model = model.train()#Put our model in training mode
        
        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training", use_tqdm = False)

        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )
        
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing", use_tqdm = False)
        
        if not use_tqdm:
            # Clear the previous output
            clear_output(wait=True)
            
            # Display the current epoch
            print(f"Completed Epoch: {epoch + 1}/{epochs}")
            
            # Display the last 5 rows of the results DataFrame
            display(pd.DataFrame(results).tail(5))
                    
    if checkpoint_file is not None:
        # Check if the directory exists, create it if not
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results' : results
            }, checkpoint_file)

    return pd.DataFrame.from_dict(results)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape) 
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)
    
class DebugShape(nn.Module):
    """
    Module that is useful to help debug your neural network architecture. 
    Insert this module between layers and it will print out the shape of 
    that layer. 
    """
    def forward(self, input):
        print(input.shape)
        return input
    
def weight_reset(m):
    """
    Go through a PyTorch module m and reset all the weights to an initial random state
    """
    if "reset_parameters" in dir(m):
        m.reset_parameters()
    return

def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj
        
def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None, score_funcs=None, 
                  epochs=50, device="cpu", checkpoint_file=None, lr_schedule=None, optimizer=None, 
                  disable_tqdm=False, resume_file=None, resume_checkpoint=False):
    """
    Train simple neural networks

    Parameters:
    model (nn.Module): The PyTorch model / "Module" to train.
    loss_func (callable): The loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score.
    train_loader (DataLoader): PyTorch DataLoader object that returns tuples of (input, label) pairs.
    val_loader (DataLoader, optional): Optional PyTorch DataLoader to evaluate on after every epoch.
    test_loader (DataLoader, optional): Optional PyTorch DataLoader to evaluate on after every epoch.
    score_funcs (dict, optional): A dictionary of scoring functions to use to evaluate the performance of the model.
    epochs (int, optional): The number of training epochs to perform (additional epochs if resuming training). Default is 50.
    device (str, optional): The compute location to perform training. Default is "cpu".
    checkpoint_file (str, optional): The file path to save the model checkpoint.
    lr_schedule (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate schedule used to alter the learning rate as the model trains. If this is not None, the user must also provide the optimizer to use.
    optimizer (torch.optim.Optimizer, optional): The method used to alter the gradients for learning.
    disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is False.
    resume_file (str, optional): The file path to the checkpoint file to resume training from. Default is None.
    resume_checkpoint (bool, optional): Whether to resume training from a checkpoint file. Default is False.

    Returns:
    DataFrame: A pandas DataFrame containing the training results.

    """
    if score_funcs is None:
        score_funcs = {}  # Empty dictionary

    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if val_loader is not None:
            to_track.append("val " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)
    if lr_schedule is not None:
        to_track.append("lr")

    total_train_time = 0  # How long have we spent in the training loop?
    start_epoch = 0
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # restore the model and optimizer state from the resume file if present
    if resume_checkpoint and resume_file is None and checkpoint_file is not None and os.path.exists(checkpoint_file):
        resume_file = checkpoint_file

    if resume_file is not None:
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Move model to the correct device
        model.to(device)

        # Now, update the optimizer's param groups with the model's parameters
        if optimizer is not None:
            # Reassign optimizer's parameter groups to match the model's current parameters
            optimizer.param_groups[0]['params'] = list(model.parameters())

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        results = checkpoint['results']
        start_epoch = checkpoint['epoch'] + 1
        total_train_time = results["total time"][-1]

        if lr_schedule is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_dict = checkpoint['lr_scheduler_state_dict']
            #lr_dict['last_epoch'] = start_epoch
            lr_schedule.load_state_dict(lr_dict)

    if optimizer is None:
        # The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Epoch", disable=disable_tqdm):

        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        if val_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode, because we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")

        # In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            results["lr"].append(optimizer.param_groups[0]['lr'])
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            elif _scheduler_accepts_epoch(lr_schedule):
                lr_schedule.step(epoch = epoch)
            else:
                lr_schedule.step()

        if test_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode, because we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")

        if checkpoint_file is not None:
            checkpoint_dir = os.path.dirname(checkpoint_file)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }
            if lr_schedule is not None:
                save_dict['lr_schedule'] = lr_schedule.state_dict()
            torch.save(save_dict, checkpoint_file)

    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)

def train_network2(model, loss_func, train_loader, val_loader=None, test_loader=None, score_funcs=None, 
                  epochs=50, device="cpu", checkpoint_file=None, lr_schedule=None, optimizer=None, 
                  disable_tqdm=False, resume_file=None, resume_checkpoint=False, early_stop=False, patience=4):
    """
    Train simple neural networks

    Parameters:
    model (nn.Module): The PyTorch model / "Module" to train.
    loss_func (callable): The loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score.
    train_loader (DataLoader): PyTorch DataLoader object that returns tuples of (input, label) pairs.
    val_loader (DataLoader, optional): Optional PyTorch DataLoader to evaluate on after every epoch.
    test_loader (DataLoader, optional): Optional PyTorch DataLoader to evaluate on after every epoch.
    score_funcs (dict, optional): A dictionary of scoring functions to use to evaluate the performance of the model.
    epochs (int, optional): The number of training epochs to perform (additional epochs if resuming training). Default is 50.
    device (str, optional): The compute location to perform training. Default is "cpu".
    checkpoint_file (str, optional): The file path to save the model checkpoint.
    lr_schedule (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate schedule used to alter the learning rate as the model trains. If this is not None, the user must also provide the optimizer to use.
    optimizer (torch.optim.Optimizer, optional): The method used to alter the gradients for learning.
    disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is False.
    resume_file (str, optional): The file path to the checkpoint file to resume training from. Default is None.
    resume_checkpoint (bool, optional): Whether to resume training from a checkpoint file. Default is False.
    early_stop (bool, optional): Whether to use early stopping. Default is False.
    patience (int, optional): The number of epochs to wait for improvement before stopping. Default is 4.

    Returns:
    DataFrame: A pandas DataFrame containing the training results.

    """
    if score_funcs is None:
        score_funcs = {}  # Empty dictionary

    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if val_loader is not None:
            to_track.append("val " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)
    if lr_schedule is not None:
        to_track.append("lr")

    total_train_time = 0  # How long have we spent in the training loop?
    start_epoch = 0
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # restore the model and optimizer state from the resume file if present
    if resume_checkpoint and resume_file is None and checkpoint_file is not None and os.path.exists(checkpoint_file):
        resume_file = checkpoint_file

    if resume_file is not None:
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Move model to the correct device
        model.to(device)

        # Now, update the optimizer's param groups with the model's parameters
        if optimizer is not None:
            # Reassign optimizer's parameter groups to match the model's current parameters
            optimizer.param_groups[0]['params'] = list(model.parameters())

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        results = checkpoint['results']
        start_epoch = checkpoint['epoch'] + 1
        total_train_time = results["total time"][-1]

        if lr_schedule is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_dict = checkpoint['lr_scheduler_state_dict']
            #lr_dict['last_epoch'] = start_epoch
            lr_schedule.load_state_dict(lr_dict)

    if optimizer is None:
        # The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    best_loss = float('inf')
    no_improvement = 0
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Epoch", disable=disable_tqdm):

        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        if val_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode, because we don't want to make any updates!
            with torch.no_grad():
                val_loss = run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
                results["val loss"].append(val_loss)

                if early_stop:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        no_improvement = 0
                        if checkpoint_file is not None:
                            save_dict = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'results': results
                            }
                            if lr_schedule is not None:
                                save_dict['lr_schedule'] = lr_schedule.state_dict()
                            torch.save(save_dict, checkpoint_file)
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            break

        # In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            results["lr"].append(optimizer.param_groups[0]['lr'])
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            elif _scheduler_accepts_epoch(lr_schedule):
                lr_schedule.step(epoch = epoch)
            else:
                lr_schedule.step()

        if test_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode, because we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")

        if checkpoint_file is not None and not early_stop:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }
            if lr_schedule is not None:
                save_dict['lr_schedule'] = lr_schedule.state_dict()
            torch.save(save_dict, checkpoint_file)

    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)

def _scheduler_accepts_epoch(scheduler):
    """
    Checks if the passed-in scheduler's step method accepts an epoch argument.
    
    Args:
        scheduler: An instance of a PyTorch learning rate scheduler.

    Returns:
        bool: True if the scheduler's step method accepts an epoch, False otherwise.
    """
    # Schedulers that accept the `epoch` argument in their step method
    schedulers_with_epoch_arg = (
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.MultiStepLR,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        torch.optim.lr_scheduler.PolynomialLR,
        torch.optim.lr_scheduler.CyclicLR
    )
    
    # Check if the scheduler is an instance of any of the above
    return isinstance(scheduler, schedulers_with_epoch_arg)

def test_network(model, loss_func, test_loader, score_funcs, device):
    """
    Test a trained regression or classification neural network on a test dataset.
    Returns the loss and scores for the model on the test set.
    Returns the predictions and ground truth values for each instance in the dataset.

    Parameters:
    model (nn.Module): The PyTorch model / "Module" to test.
    loss_func (callable): The loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score.
    test_loader (DataLoader): PyTorch DataLoader object that returns tuples of (input, label) pairs.
    score_funcs (dict): A dictionary of scoring functions to use to evaluate the performance of the model.
    device (torch.device): The device to move the model and tensors to.

    Returns:
    DataFrame: A pandas DataFrame containing the loss function value and metrics across the whole test set.
    DataFrame: A pandas DataFrame containing the predictions and ground truth for each instance in the dataset.

    """
    model = model.to(device)  # Move the model to the device
    model = model.eval()  # Set the model to "evaluation" mode, because we don't want to make any updates!
    results = {}
    predictions = []
    ground_truth = []
    loss_accumulated = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # Move the inputs to the device
            labels = labels.to(device)  # Move the labels to the device

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss_accumulated += loss.item()

            # Convert predictions to labels (if classification problem)
            outputs = np.asarray(outputs)
            if len(outputs.shape) == 2 and outputs.shape[1] > 1:  # If classification, convert to labels
                outputs = np.argmax(outputs, axis=1)

            for score_name, score_func in score_funcs.items():
                score = score_func(outputs, labels)
                results[score_name] = results.get(score_name, 0) + score

            predictions.extend(outputs.tolist())
            ground_truth.extend(labels.tolist())

        # Calculate average scores
        for score_name in score_funcs.keys():
            results[score_name] /= len(test_loader)

        loss_average = loss_accumulated / len(test_loader)
        results['loss'] = loss_average

    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
    df_preds = pd.DataFrame({'predictions': predictions, 'ground_truth': ground_truth})

    return df_results, df_preds


def check_and_install_packages(packages):
    """
    Check if the given packages are installed, and if not, install them.
    
    :param packages: List of package names to check and install if necessary.
    """
    environment = detect_jupyter_environment()
    
    for package_name in packages:
        try:
            importlib.import_module(package_name)
            print(f"'{package_name}' is already installed.")
        except ImportError:
            print(f"'{package_name}' is not installed. Installing now...")
            if environment == "cocalc":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"'{package_name}' has been installed.")

###########################################################
# RNN utility Classes 
###########################################################

class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1    
    
    def forward(self, input):
        #Result is either a tupe (out, h_t)
        #or a tuple (out, (h_t, c_t))
        rnn_output = input[0]

        last_step = input[1]
        if(type(last_step) == tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        #We want the last layer's results
        last_step = last_step[self.rnn_layers-1] 
        #Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        #Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)
    
class EmbeddingPackable(nn.Module):
    """
    The embedding layer in PyTorch does not support Packed Sequence objects. 
    This wrapper class will fix that. If a normal input comes in, it will 
    use the regular Embedding layer. Otherwise, it will work on the packed 
    sequence to return a new Packed sequence of the appropriate result. 
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer 
    
    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input, 
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            #And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), 
                                                           batch_first=True, enforce_sorted=False)
        else:#apply to normal data
            return self.embd_layer(input)

### Attention Mechanism Layers

class ApplyAttention(nn.Module):
    """
    This helper module is used to apply the results of an attention mechanism toa set of inputs. 
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()
        
    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape 
            (B, T), with `True` indicating which items are present / valid. 
            
        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1). 
        """
        
        if mask is not None:
            #set everything not present to a large negative value that will cause vanishing gradients 
            attention_scores[~mask] = -1000.0
        #compute the weight for each score
        weights = F.softmax(attention_scores, dim=1) #(B, T, 1) still, but sum(T) = 1
    
        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights

class AttentionAvg(nn.Module):

    def __init__(self, attnScore):
        super(AttentionAvg, self).__init__()
        self.score = attnScore
    
    def forward(self, states, context, mask=None):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, D), a weighted av
        
        """
        
        B = states.size(0)
        T = states.size(1)
        D = states.size(2)
        
        scores = self.score(states, context) #(B, T, 1)
        
        if mask is not None:
            scores[~mask] = float(-10000)
        weights = F.softmax(scores, dim=1) #(B, T, 1) still, but sum(T) = 1
        
        context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D, 1)
        
        
        return context.view(B, D) #Flatten this out to (B, D)


class AdditiveAttentionScore(nn.Module):

    def __init__(self, D):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(D, 1)
        self.w = nn.Linear(2*D, D)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, D) + (B, T, D)  -> (B, T, 2*D)
        scores = self.v(torch.tanh(self.w(state_context_combined)))
        return scores

class GeneralScore(nn.Module):

    def __init__(self, D):
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(D, D, 1)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        scores = self.w(states, context) #(B, T, D) -> (B, T, 1)
        return scores

class DotScore(nn.Module):

    def __init__(self, D):
        super(DotScore, self).__init__()
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        
        scores = torch.bmm(states,context.unsqueeze(2)) / np.sqrt(D) #(B, T, D) -> (B, T, 1)
        return scores
    
def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: the original input with three or more dimensions, (B, ..., T, ...)
        which may have unsued items in the tensor. B is the batch size, 
        and T is the time dimension. 
    time_dimension: the axis in the tensor `x` that denotes the time dimension
    fill: the constant used to denote that an item in the tensor is not in use,
        and should be masked out (`False` in the mask). 
    
    return: A boolean tensor of shape (B, T), where `True` indicates the value
        at that time is good to use, and `False` that it is not. 
    """
    to_sum_over = list(range(1,len(x.shape))) #skip the first dimension 0 because that is the batch dimension
    
    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)
       
    with torch.no_grad():
        #Special case is when shape is (B, D), then it is an embedding layer. We just return the values that are good
        if len(to_sum_over) == 0:
            return (x != fill)
        #(x!=fill) determines locations that might be unused, beause they are 
        #missing the fill value we are looking for to indicate lack of use. 
        #We then count the number of non-fill values over everything in that
        #time slot (reducing changes the shape to (B, T)). If any one entry 
        #is non equal to this value, the item represent must be in use - 
        #so return a value of true. 
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask

class LanguageNameDataset(Dataset):
    
    def __init__(self, lang_name_dict, vocabulary):
        self.label_names = [x for x in lang_name_dict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.label_names):
            for sample in lang_name_dict[language]:
                self.data.append(sample)
                self.labels.append(y)
        
    def __len__(self):
        return len(self.data)
    
    def _string2InputVec(self, input_string):
        """
        This method will convert any input string into a vector of long values, according to the vocabulary used by this object. 
        input_string: the string to convert to a tensor
        """
        T = len(input_string) #How many characters long is the string?
        
        #Create a new tensor to store the result in
        name_vec = torch.zeros((T), dtype=torch.long)
        #iterate through the string and place the appropriate values into the tensor
        for pos, character in enumerate(input_string):
            name_vec[pos] = self.vocabulary[character]
            
        return name_vec
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        
        #Conver the correct class label into a tensor for PyTorch
        label_vec = torch.tensor([label], dtype=torch.long)
        
        return self._string2InputVec(name), label
    
def pad_and_pack(batch):
    #1, 2, & 3: organize the batch input lengths, inputs, and outputs as seperate lists
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)
    #4: create the padded version of the input
    x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    #5: create the packed version from the padded & lengths
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    #Convert the lengths into a tensor
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    #6: return a tuple of the packed inputs and their labels
    return x_packed, y_batched