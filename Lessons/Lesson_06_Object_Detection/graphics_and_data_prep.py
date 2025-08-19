import os
import re
import cv2
import random
import shutil
import zipfile
import numpy as np
import torch
from pathlib import Path
from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
import requests
from torchvision import ops
from torchvision.io import read_image
import ipywidgets as widgets
from IPython.display import display, clear_output
import ipywidgets as widgets
import json
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
import yaml
import ipywidgets as widgets
from ipywidgets import interact, widgets
from introdl.idlmam import save_checkpoint
from introdl.utils import load_model, load_results
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm.autonotebook")
from tqdm.autonotebook import tqdm

def train_one_epoch(model, optimizer, train_loader, val_loader, device, map_metric, scheduler, step_per_batch=False):
    '''
    Trains the model for one epoch and evaluates it on the validation set.
    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        map_metric (torchmetrics.Metric): Metric for calculating mean Average Precision (mAP).
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        step_per_batch (bool, optional): Whether to step the scheduler per batch. Default is False.
    Returns:
        tuple: A tuple containing:
            - avg_train_loss (float): The average training loss for the epoch.
            - avg_val_loss (float): The average validation loss for the epoch.
            - map_50 (float): The mAP at IoU=0.50.
            - map_50_95 (float): The mAP at IoU=0.50:0.95.
    '''

    model.train()
    running_loss = 0.0
    
    # Training phase
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += losses.item()

        if scheduler and step_per_batch:
            scheduler.step()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()  # Switch to eval mode to get predictions
    val_running_loss = 0.0
    map_metric.reset()  # Reset metrics before validation

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Get model predictions without targets to avoid returning losses
            outputs = model(images)
            
            # Format outputs and targets for torchmetrics
            formatted_outputs = [
                {k: v.cpu() for k, v in output.items()} for output in outputs
            ]
            formatted_targets = [
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in target.items()} for target in targets
            ]
            
            # Update mAP metric with formatted outputs and targets
            map_metric.update(formatted_outputs, formatted_targets)
            
            # Temporarily switch to training mode to compute validation loss
            model.train()
            loss_dict = model(images, targets)
            val_losses = sum(loss for loss in loss_dict.values())
            val_running_loss += val_losses.item()
            
            # Switch back to eval mode for the next prediction
            model.eval()

    # Calculate mAP metrics
    avg_val_loss = val_running_loss / len(val_loader)
    map_metrics = map_metric.compute()
    map_50 = map_metrics['map_50']
    map_50_95 = map_metrics['map']
    
    return avg_train_loss, avg_val_loss, map_50, map_50_95

def train_rcnn(model, optimizer, scheduler, train_loader, val_loader, device, checkpoint_file, 
               num_epochs=15, step_per_batch=False, patience=None, pretend_train=False):
    '''
    Train a Region-based Convolutional Neural Network (RCNN) model.

    Args:
        model (torch.nn.Module): The RCNN model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        checkpoint_file (str): Path to save the final model weights, results, etc.
        num_epochs (int, optional): Number of epochs to train the model. Default is 15.
        step_per_batch (bool, optional): Whether to step the scheduler per batch. Default is False.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is None.
        pretend_train (bool, optional): If True, simulate training if a checkpoint file is available. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing training and validation metrics for each epoch.
    '''

    print(pretend_train, checkpoint_file)

    # Pretend training
    if pretend_train and checkpoint_file is not None:
        try:
            print("Attempting to load model and results for pretend training...")

            # Load the model
            model = load_model(model, checkpoint_file, device=device)
            print("Model loaded successfully.")

            # Load results
            results = load_results(checkpoint_file, device=device)
            print("Results loaded successfully. Simulating training...")

            # Simulate training with progress bar
            import time
            with tqdm(total=num_epochs, desc="Pretend Training", dynamic_ncols=True) as pbar:
                for _ in range(num_epochs):
                    time.sleep(5 / num_epochs)  # Simulate all epochs in 5 seconds
                    pbar.update(1)

            return results  # Return the actual results loaded from the checkpoint

        except FileNotFoundError:
            print(f"Checkpoint file {checkpoint_file} not found. Proceeding with actual training...")
        except Exception as e:
            print(f"Error during pretend training: {e}. Proceeding with actual training...")

    # Initialize mAP metrics for validation
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

    # DataFrame to store results
    results = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'map_50', 'map_50_95', 'lr'])

    best_map_50_95 = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Train and validate for the epoch
        avg_train_loss, avg_val_loss, map_50, map_50_95 = train_one_epoch(
            model, optimizer, train_loader, val_loader, device, map_metric, scheduler, step_per_batch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Append results to DataFrame
        results.loc[len(results)] = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'map_50': map_50.item(),
            'map_50_95': map_50_95.item(),
            'lr': current_lr
        }

        print((f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
               f"Val Loss: {avg_val_loss:.4f}, mAP@50: {map_50:.4f}, mAP@50-95: {map_50_95:.4f}, LR: {current_lr:.6f}"))

        # Check for improvement and save the best model if patience is specified
        if patience is not None:
            if map_50_95 > best_map_50_95:
                best_map_50_95 = map_50_95
                epochs_without_improvement = 0
                save_checkpoint(epoch, model, optimizer, results, checkpoint_file, scheduler)
                print(f"Improvement detected. Model weights saved.")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break

        # Step the scheduler based on mAP@50-95
        if scheduler and not step_per_batch:
            scheduler.step(map_50)

    # Save the model from the last epoch if early stopping is not triggered
    if patience is None or epochs_without_improvement < patience:
        save_checkpoint(epoch, model, optimizer, results, checkpoint_file, scheduler)
        print(f"Model weights saved from the last epoch.")

    print(f"Training complete. Best model weights saved at epoch {epoch + 1 - epochs_without_improvement}.")
    return results

'''
def train_rcnn(model, optimizer, scheduler, train_loader, val_loader, device, save_file, 
               num_epochs=15, step_per_batch=False, patience=None):
    """
    Train a Region-based Convolutional Neural Network (RCNN) model.
    Args:
        model (torch.nn.Module): The RCNN model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        save_file (str): Path to save the final model weights.
        num_epochs (int, optional): Number of epochs to train the model. Default is 15.
        step_per_batch (bool, optional): Whether to step the scheduler per batch. Default is False.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is None.
    Returns:
        pd.DataFrame: DataFrame containing training and validation metrics for each epoch.
    """

    # Initialize mAP metrics for validation
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

    # DataFrame to store results
    results = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'map_50', 'map_50_95', 'lr'])

    best_map_50_95 = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Train and validate for the epoch
        avg_train_loss, avg_val_loss, map_50, map_50_95 = train_one_epoch(model, optimizer, train_loader, 
                                                                          val_loader, device, map_metric, 
                                                                          scheduler, step_per_batch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Append results to DataFrame
        results.loc[len(results)] = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'map_50': map_50.item(),
            'map_50_95': map_50_95.item(),
            'lr': current_lr
        }

        epoch_str = (f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                     f"Val Loss: {avg_val_loss:.4f}, mAP@50: {map_50:.4f}, mAP@50-95: {map_50_95:.4f}, LR: {current_lr:.6f}")

        print(epoch_str)

        # Check for improvement and save the best model if patience is specified
        if patience is not None:
            if map_50_95 > best_map_50_95:
                best_map_50_95 = map_50_95
                epochs_without_improvement = 0
                # Save the best model weights
                torch.save(model.state_dict(), save_file)
                print(f"Improvement detected. Model weights saved.")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break

        # Step the scheduler based on mAP@50-95
        if scheduler and not step_per_batch:
            scheduler.step(map_50)

    # Save the model from the last epoch if early stopping is not triggered
    if patience is None or epochs_without_improvement < patience:
        torch.save(model.state_dict(), save_file)
        print(f"Model weights saved from the last epoch.")

    print(f"Training complete. Best model weights saved at epoch {epoch + 1 - epochs_without_improvement}.")

    return results
'''
    
def download_pennfudanped(target_dir: Path) -> None:
    """
    Downloads and unzips the PennFudanPed dataset into the specified directory.
    Args:
        target_dir (Path): The directory where the dataset will be downloaded and unzipped.
    Returns:
        None
    """
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = target_dir / "PennFudanPed.zip"
    dataset_path = target_dir / "PennFudanPed"

    if dataset_path.exists():
        print("PennFudanPed dataset already exists.")
        return
    
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PennFudanPed dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download complete.")

    print("Unzipping the dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print("Unzipping complete.")
    zip_path.unlink()
    print("Zip file deleted. PennFudanPed dataset is ready.")


def prepare_penn_fudan_yolo(target_dir: Path, seed=42):
    # Define paths for the dataset and processed directories
    dataset_dir = target_dir / "PennFudanPed"
    yolo_dataset_dir = target_dir / "PennFudanPedYOLO"

    # Step 1: Check if PennFudanPedYOLO already exists
    if yolo_dataset_dir.exists():
        print("PennFudanPedYOLO already exists. Stopping preparation.")
        return

    # Step 2: Download and unzip the dataset if necessary
    download_pennfudanped(target_dir)

    # Step 3: Create the YOLO directory structure
    images_dir = dataset_dir / "PNGImages"
    annotations_dir = dataset_dir / "Annotation"
    labels_dir = yolo_dataset_dir / "labels"
    train_images_dir = yolo_dataset_dir / "images" / "train"
    val_images_dir = yolo_dataset_dir / "images" / "val"
    train_labels_dir = yolo_dataset_dir / "labels" / "train"
    val_labels_dir = yolo_dataset_dir / "labels" / "val"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Convert annotations to YOLO format
    def convert_annotation_to_yolo(annotation_file, image_width, image_height):
        yolo_annotations = []

        with annotation_file.open('r') as file:
            lines = file.readlines()

            for line in lines:
                if "Bounding box" in line:
                    # Extract coordinates using regular expressions
                    match = re.search(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", line)
                    if match:
                        Xmin, Ymin, Xmax, Ymax = map(int, match.groups())

                        # Convert to YOLO format
                        x_center = (Xmin + Xmax) / 2 / image_width
                        y_center = (Ymin + Ymax) / 2 / image_height
                        width = (Xmax - Xmin) / image_width
                        height = (Ymax - Ymin) / image_height
                        yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}\n")

        return yolo_annotations

    # Step 5: Process all annotation files and split dataset
    random.seed(seed)
    image_files = sorted([f for f in images_dir.glob("*.png")])
    random.shuffle(image_files)

    split_index = int(0.8 * len(image_files))
    train_files = {f.stem for f in image_files[:split_index]}  # Use base filenames for faster lookups
    val_files = {f.stem for f in image_files[split_index:]}

    for image_path in image_files:
        # Read image dimensions
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]

        # Convert annotation to YOLO format
        annotation_file = annotations_dir / (image_path.stem + ".txt")
        if annotation_file.exists():
            yolo_data = convert_annotation_to_yolo(annotation_file, w, h)

            # Save YOLO annotation
            label_dir = train_labels_dir if image_path.stem in train_files else val_labels_dir
            yolo_label_path = label_dir / (image_path.stem + ".txt")
            with yolo_label_path.open('w') as f:
                f.writelines(yolo_data)

        # Move the image to the appropriate folder
        image_dir = train_images_dir if image_path.stem in train_files else val_images_dir
        shutil.copy(image_path, image_dir / image_path.name)

    # Step 6: Create dataset.yaml file
    yaml_file = yolo_dataset_dir / "dataset.yaml"
    yaml_data = {
        "train": str(train_images_dir),
        "val": str(val_images_dir),
        "nc": 1,
        "names": ["person"]
    }

    with yaml_file.open('w') as f:
        yaml.dump(yaml_data, f)

    print("Dataset preparation complete.")

def overlay_masks(image, mask, color, thickness=1, mode="shaded", alpha=0.3):
    """
    Overlay a binary mask outline on an image.

    Parameters:
    - image (numpy.ndarray): The image on which to overlay the mask.
    - mask (numpy.ndarray): The binary mask (1 for mask, 0 for background).
    - color (tuple): The color for the mask outline in (B, G, R) format.
    - thickness (int, optional): The thickness of the mask outline (default: 1).
    - mode (str, optional): The mode of overlay ("outline", "shaded", "both") (default: "shaded").
    - alpha (float, optional): The transparency level for shaded mode (default: 0.3).

    Returns:
    - image (numpy.ndarray): The image with the mask outline overlaid.
    """
    mask = (mask > 0.5).astype(np.uint8)
    if len(mask.shape) > 2:
        mask = mask.squeeze()  # Ensure single-channel mask

    if mode in ["shaded", "both"]:
        overlay = image.copy()
        overlay[mask == 1] = color
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    if mode in ["outline", "both"]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, thickness)

    return image

def denormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Denormalizes an image by applying the given mean and standard deviation.
    Parameters:
    image (np.ndarray): The normalized image array.
    mean (list[float]): The mean values used for normalization.
    std (list[float]): The standard deviation values used for normalization.
    Returns:
    np.ndarray: The denormalized image array, with values clipped between 0 and 1.
    """
    mean = np.array(mean)
    std = np.array(std)
    return (image * std + mean).clip(0, 1)

def display_images_and_masks(dataset, num_samples=3, model=None, indices=None, figsize=(4, 4), overlay=True, 
                             denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode="outline", alpha=0.3):
    """
    Display random samples from the dataset with ground truth masks. If a model is provided, it also displays predicted masks.

    Parameters:
    - dataset: The dataset containing the images and ground truth masks.
    - num_samples: The number of random samples to display (default: 3).
    - model: The segmentation model. If None, only ground truth masks are displayed.
    - indices (list, optional): The indices of the samples to display. If None, random samples will be selected.
    - figsize (tuple, optional): The size of the display figure (default: (6, 6)).
    - overlay (bool, optional): If True, overlays the mask on the image. If False, displays image and mask side-by-side.
    - denormalize (bool, optional): If True, denormalizes the images before displaying (default: False).
    """
    if indices is None:
        indices = random.sample(range(len(dataset)), num_samples)
    
    device = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device

    for idx in indices:
        # Load the image and ground truth mask
        image, target_mask = dataset[idx]
        
        # Move image to device for model prediction
        pred_mask = None
        if model is not None:
            image_for_model = image.to(device)
            with torch.no_grad():
                logits = model(image_for_model.unsqueeze(0))[0].squeeze().cpu()
                pred_mask = torch.sigmoid(logits.clone().detach()).numpy()

        # Convert image to numpy and optionally denormalize
        img_np = image.permute(1, 2, 0).cpu().numpy() if device else image.permute(1, 2, 0).numpy()
        if denormalize:
            img_np = denormalize_image(img_np, mean, std)
        img_np = (img_np * 255).astype(np.uint8)  # Scale for OpenCV display

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gt_mask = target_mask.cpu().numpy() if device else target_mask.numpy()

        if overlay:

            if pred_mask is None:
                # Overlay ground truth mask outline in cyan
                img_overlay = overlay_masks(img_bgr.copy(), gt_mask, color=(255, 255, 0), mode=mode, alpha=alpha)
            else:
                # Overlay predicted mask in orange with cyan outline for ground truth
                pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize predicted mask
                img_overlay = overlay_masks(img_bgr.copy(), pred_mask, color=(0, 165, 255), mode=mode, alpha=alpha)   
                img_overlay = overlay_masks(img_overlay, gt_mask, color=(255, 255, 0), mode="outline", alpha=1.0)

            img_rgb_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb_overlay)
            plt.axis('off')
            plt.show()

        else:
            # Prepare separate images for side-by-side display
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_rgb = (gt_mask.squeeze() * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_rgb] * 3, axis=-1)  # Convert to 3-channel grayscale
            
            images = [img_rgb, mask_rgb]
            titles = ['Image', 'Ground Truth Mask']
            
            # Add predicted mask if model is provided
            if pred_mask is not None:
                pred_rgb = (pred_mask * 255).astype(np.uint8)  # Display as binary mask in grayscale
                pred_rgb = np.stack([pred_rgb] * 3, axis=-1)  # Convert to 3-channel grayscale
                if mode=="outline":
                    pred_rbg = overlay_masks(pred_rgb, gt_mask, color=(0, 255, 255), mode="outline", alpha=1.0)
                    pred_rgb = overlay_masks(pred_rgb, pred_mask, color=(255, 165, 0), mode="outline", alpha=1.0)
                images.append(pred_rgb)
                titles.append('Predicted Mask')
            
            plt.figure(figsize=(figsize[0] * len(images), figsize[1]))
            for i, (title, img) in enumerate(zip(titles, images)):
                plt.subplot(1, len(images), i + 1)
                plt.imshow(img)
                plt.title(title)
                plt.axis('off')
            plt.show()


def display_boxes(ax, boxes, color, label):
    """
    Display bounding boxes on an axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to display the bounding boxes.
    boxes (numpy.ndarray): The bounding boxes to be displayed. Each box should have four coordinates (x, y, xmax, ymax).
    color (str): The color of the bounding boxes.
    label (str): The label for the bounding boxes.

    Returns:
    None
    """
    boxes = np.atleast_2d(boxes)  # Ensure boxes are 2D
    for box in boxes:
        if len(box) == 4:  # Check if box has four coordinates
            x, y, xmax, ymax = box
            width, height = xmax - x, ymax - y
            rect = Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none', label=label)
            ax.add_patch(rect)

# Function to denormalize the image tensor for display
def denormalize_torch(image_tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """
    Denormalizes a PyTorch tensor image using the provided mean and standard deviation.

    Args:
        image_tensor (torch.Tensor): The normalized image tensor to be denormalized.
        mean (list[float]): The mean values used for normalization.
        std (list[float]): The standard deviation values used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return image_tensor * std[:, None, None] + mean[:, None, None]

def display_images_and_boxes(dataset, num_samples=3, model=None, min_confidence=0.5, iou_max_overlap=0.4, indices=None, 
                             figsize=(6, 6), denormalize=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                             show_confidence=False, show_prediction=False, filter_mode='nms'):
    """
    Display random samples from the dataset with ground truth boxes. If a model is provided, it also displays predictions.

    Parameters:
    - dataset: The dataset containing the images and ground truth annotations.
    - num_samples: The number of random samples to display (default: 3).
    - model: The object detection model. If None, only ground truth boxes are displayed.
    - min_confidence: The confidence threshold for filtering predictions (default: 0.5).
    - iou_max_overlap: The IoU threshold for Non-Max Suppression (default: 0.4).
    - indices (list, optional): The indices of the samples to display. If None, random samples will be selected.
    - figsize (tuple, optional): The size of the figure (default: (6,6)).
    - denormalize (bool): Whether to denormalize the image for display.
    - show_confidence (bool): Whether to show confidence scores on predicted boxes.
    - show_prediction (bool): Whether to show predicted class labels on boxes.
    - filter_mode (str): The mode for filtering predictions ('nms' for Non-Max Suppression, 'gt' for ground truth filtering).
    """
    if indices is None:
        indices = random.sample(range(len(dataset)), num_samples)
    
    device = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device

    for idx in indices:
        # Load the image and target directly from the dataset
        image, target = dataset[idx]
        
        # Denormalize the image if needed
        if denormalize:
            image = denormalize_torch(image, mean, std)
        
        img_np = image.permute(1, 2, 0).cpu().numpy() if device else image.permute(1, 2, 0).numpy()
        
        # Prepare the figure
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img_np)
        
        # Check if ground truth boxes are available
        if "boxes" in target and target["boxes"].numel() > 0:
            gt_boxes = target["boxes"].cpu().numpy() if device else target["boxes"].numpy()
            display_boxes(ax, gt_boxes, color="blue", label="Ground Truth")
        else:
            print(f"No ground truth boxes found for index {idx}")

        if model is not None:
            # Move image to device if model is specified
            image = image.to(device)
            with torch.no_grad():
                output = model([image])[0]
            
            # Filter predictions by confidence threshold
            pred_boxes = output["boxes"].cpu().detach().numpy()
            pred_scores = output["scores"].cpu().detach().numpy()
            pred_labels = output["labels"].cpu().detach().numpy()
            
            # Apply confidence threshold
            relevant_boxes = pred_boxes[pred_scores > min_confidence]
            relevant_scores = pred_scores[pred_scores > min_confidence]
            relevant_labels = pred_labels[pred_scores > min_confidence]

            if filter_mode == 'nms':
                # Apply Non-Maximum Suppression if needed
                if iou_max_overlap < 1.0 and len(relevant_boxes) > 0:
                    keep = ops.nms(torch.tensor(relevant_boxes), torch.tensor(relevant_scores), iou_max_overlap)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]
            elif filter_mode == 'gt':
                # Filter out predictions with IoU below threshold with any ground truth box
                if len(relevant_boxes) > 0 and len(gt_boxes) > 0:
                    ious = ops.box_iou(torch.tensor(relevant_boxes), torch.tensor(gt_boxes)).numpy()
                    keep = np.any(ious > iou_max_overlap, axis=1)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]

            # Ensure they are arrays even if there's only one relevant box
            if relevant_boxes.ndim == 1:
                relevant_boxes = np.expand_dims(relevant_boxes, axis=0)
                relevant_scores = np.expand_dims(relevant_scores, axis=0)
                relevant_labels = np.expand_dims(relevant_labels, axis=0)
            
            # Display predicted boxes in orange
            display_boxes(ax, relevant_boxes, color="orange", label="Prediction")
            
            # Show confidence scores and predicted class labels if required
            if show_confidence or show_prediction:
                class_names = getattr(dataset, 'classes', None)
                for box, score, label in zip(relevant_boxes, relevant_scores, relevant_labels):
                    x, y, _, _ = box
                    text = ""
                    if show_confidence:
                        text += f"{score:.2f}"
                    if show_prediction:
                        if class_names:
                            text += f", {class_names[label]}"
                        else:
                            text += f", {label}"
                    ax.text(x, y, text, color="white", fontsize=8, bbox=dict(facecolor="orange", alpha=0.5))
            
            # Set up legend with custom labels for ground truth and predictions
            handles = [
                Rectangle((0, 0), 1, 1, edgecolor="blue", facecolor='none', label="Ground Truth"),
                Rectangle((0, 0), 1, 1, edgecolor="orange", facecolor='none', label="Prediction")
            ]
        else:
            # Only ground truth legend
            handles = [Rectangle((0, 0), 1, 1, edgecolor="blue", facecolor='none', label="Ground Truth")]
        
        plt.legend(handles=handles)
        plt.axis('off')
        plt.show()


def display_yolo_predictions(yaml_file, model, num_samples=3, imgsz=640, conf=0.25,
                              indices=None, figsize=(6, 6), show_confidence=True, max_iou_threshold=0.5):
    """
    Display random samples from a dataset defined in the dataset.yaml file with predictions from a fine-tuned YOLO model.
    Predictions are processed using the same resizing and postprocessing pipeline as model.val.

    Parameters:
    - yaml_file: Path to the dataset.yaml file containing dataset configuration.
    - model: A YOLO model instance.
    - num_samples: The number of random samples to display (default: 3).
    - imgsz: Image size to resize for inference (default: 640).
    - conf: Confidence threshold for predictions (default: 0.25).
    - indices (list, optional): The indices of the samples to display. If None, random samples will be selected.
    - figsize (tuple, optional): The size of the figure (default: (6,6)).
    - show_confidence (bool): Whether to show confidence scores on predicted boxes.
    - max_iou_threshold (float): Maximum IoU threshold for NMS filtering (default: 0.5).
    """
    # Load the dataset from the yaml file
    with open(yaml_file, 'r') as file:
        dataset_info = yaml.safe_load(file)

    val_images_path = dataset_info.get('val', [])
    if isinstance(val_images_path, str):
        # If 'val' is a directory, get all image paths
        val_images_path = [os.path.join(val_images_path, fname) for fname in os.listdir(val_images_path)
                           if fname.endswith(('.jpg', '.png', '.jpeg'))]

    if indices is None:
        indices = random.sample(range(len(val_images_path)), num_samples)

    for idx in indices:
        # Get the image path
        image_path = val_images_path[idx]

        # Perform prediction on the image (mimics model.val)
        results = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=max_iou_threshold, verbose=False)
        predictions = results[0].boxes  # Postprocessed predictions
        pred_boxes = predictions.xyxy.cpu().numpy()  # Bounding box coordinates
        pred_scores = predictions.conf.cpu().numpy()  # Confidence scores

        # Load the image for visualization
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Load ground truth boxes
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        with open(label_path, 'r') as file:
            gt_boxes = []
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width / 2) * img_array.shape[1]
                y1 = (y_center - height / 2) * img_array.shape[0]
                x2 = (x_center + width / 2) * img_array.shape[1]
                y2 = (y_center + height / 2) * img_array.shape[0]
                gt_boxes.append([x1, y1, x2, y2])
            gt_boxes = np.array(gt_boxes)

        # Visualize the predictions and ground truth
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img_array)

        # Plot predicted boxes in orange
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
            if show_confidence:
                ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=8, bbox=dict(facecolor='orange', alpha=0.7))

        # Plot ground truth boxes in blue
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # Add legend
        handles = [Rectangle((0, 0), 1, 1, edgecolor="orange", facecolor='none', label="Prediction"),
                   Rectangle((0, 0), 1, 1, edgecolor="blue", facecolor='none', label="Ground Truth")]
        ax.legend(handles=handles)
        ax.axis('off')
        plt.show()


def mAP_widget():
    # Load image and annotations
    image = read_image("./pictures/stick_peds.png")
    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores = load_annotations("./pictures/stick_peds.json")

    # Display image with interactive sliders and adjustable font sizes
    display_image_with_dropdown(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels)

def display_image_with_dropdown(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, 
                                detection_iou=0.5, conf_threshold=1.0, image_width_scale=0.8,
                                box_font_size=12, stats_font_size=14, pr_curve_font_size=12,
                                scores_with_background=False, iou_max_overlap=1.0):

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Define slider layouts and styles
    slider_layout = widgets.Layout(width='500px')
    slider_style = {'description_width': '150px'}

    # Create sliders
    iou_slider = widgets.FloatSlider(
        min=0.05, max=0.95, step=0.05, value=detection_iou, 
        description='Detection IoU:', layout=slider_layout, style=slider_style
    )
    conf_slider = widgets.FloatSlider(
        min=0, max=1, step=0.05, value=conf_threshold, 
        description='Min Confidence:', layout=slider_layout, style=slider_style
    )
    nms_slider = widgets.FloatSlider(
        min=0.0, max=1.0, step=0.05, value=iou_max_overlap, 
        description='NMS IOU Max Overlap:', layout=slider_layout, style=slider_style
    )
    output = widgets.Output()

    def update_display():
        with output:
            output.clear_output(wait=True)
            display_image_with_boxes(
                image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
                iou_slider.value, conf_slider.value, image_width_scale,
                box_font_size, stats_font_size, pr_curve_font_size,
                scores_with_background, nms_slider.value
            )

    # Call update_display() once to show initial plots
    update_display()

    # Attach observers for updates
    iou_slider.observe(lambda change: update_display(), names='value')
    conf_slider.observe(lambda change: update_display(), names='value')
    nms_slider.observe(lambda change: update_display(), names='value')

    # Display widget
    display(widgets.VBox([iou_slider, conf_slider, nms_slider, output]))



def calculate_iou(box, gt_boxes):
    """
    Calculate the IoU (Intersection over Union) between a predicted box and a list of ground truth boxes.
    """
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height

    gt_x_min = gt_boxes[:, 0]
    gt_y_min = gt_boxes[:, 1]
    gt_x_max = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_y_max = gt_boxes[:, 1] + gt_boxes[:, 3]

    inter_x_min = np.maximum(x_min, gt_x_min)
    inter_y_min = np.maximum(y_min, gt_y_min)
    inter_x_max = np.minimum(x_max, gt_x_max)
    inter_y_max = np.minimum(y_max, gt_y_max)

    inter_width = np.maximum(0, inter_x_max - inter_x_min)
    inter_height = np.maximum(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    box_area = width * height
    gt_areas = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)
    union = box_area + gt_areas - intersection

    return intersection / union

def compute_precisions_recalls_mAP(adjusted_pred_boxes, pred_scores, pred_labels, adjusted_gt_boxes, gt_labels, confidence_thresholds, detection_iou):
    """
    Compute precisions, recalls, and mAP for a given IoU threshold.
    """
    precisions = []
    recalls = []
    for conf in confidence_thresholds:
        tp, fp = 0, 0
        matched_gt_indices_conf = set()
        for box, score, label in zip(adjusted_pred_boxes, pred_scores, pred_labels):
            if score >= conf:
                matching_indices = np.where(gt_labels == label)[0]
                matching_gt_boxes = adjusted_gt_boxes[matching_indices]
                max_iou = 0
                max_iou_idx = -1
                if len(matching_gt_boxes) > 0:
                    iou_values = calculate_iou(box, matching_gt_boxes)
                    max_iou_idx = np.argmax(iou_values)
                    max_iou = iou_values[max_iou_idx]
                    matched_gt_idx = matching_indices[max_iou_idx]
                    if max_iou >= detection_iou and matched_gt_idx not in matched_gt_indices_conf:
                        tp += 1
                        matched_gt_indices_conf.add(matched_gt_idx)
                    else:
                        fp += 1
                else:
                    fp += 1
        fn = len(adjusted_gt_boxes) - len(matched_gt_indices_conf)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    # Store the original precisions and recalls before adjustment
    original_precisions = precisions.copy()
    original_recalls = recalls.copy()
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    # Ensure precision is non-increasing
    adjusted_precisions = precisions.copy()
    for i in range(len(adjusted_precisions) - 2, -1, -1):
        adjusted_precisions[i] = max(adjusted_precisions[i], adjusted_precisions[i + 1])
    # Extend the arrays to cover Recall from 0 to 1
    extended_recalls = np.concatenate(([0], recalls, [1]))
    extended_precisions = np.concatenate(([adjusted_precisions[0]], adjusted_precisions, [0]))
    # Compute mAP as area under the adjusted precision-recall curve
    mAP = np.trapz(extended_precisions, extended_recalls)
    return mAP, original_precisions, original_recalls, adjusted_precisions, recalls, extended_precisions, extended_recalls

def compute_mAP(adjusted_pred_boxes, pred_scores, pred_labels, adjusted_gt_boxes, gt_labels, confidence_thresholds, detection_iou):
    """
    Compute mAP for a given IoU threshold.
    """
    mAP, _, _, _, _, _, _ = compute_precisions_recalls_mAP(adjusted_pred_boxes, pred_scores, pred_labels,
                                                            adjusted_gt_boxes, gt_labels, confidence_thresholds, detection_iou)
    return mAP

def display_image_with_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, 
                             detection_iou, conf_threshold, image_width_scale,
                             box_font_size, stats_font_size, pr_curve_font_size,
                             scores_with_background, nms_threshold):
    """
    Display an image with ground truth and predicted bounding boxes, apply color coding based on thresholds,
    display precision and recall, and plot the precision-recall curve.
    """
    from PIL import Image

    # Resize the image
    original_height, original_width, _ = image.shape
    new_width = int(original_width * image_width_scale)
    new_height = int(original_height * image_width_scale)
    resized_image = np.array(Image.fromarray(image).resize((new_width, new_height), Image.LANCZOS))

    # Scale the bounding boxes accordingly
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    adjusted_gt_boxes = gt_boxes.copy()
    adjusted_gt_boxes[:, [0, 2]] *= width_scale  # x and width
    adjusted_gt_boxes[:, [1, 3]] *= height_scale  # y and height

    adjusted_pred_boxes = pred_boxes.copy()
    adjusted_pred_boxes[:, [0, 2]] *= width_scale
    adjusted_pred_boxes[:, [1, 3]] *= height_scale

    # Convert adjusted_pred_boxes from (x, y, w, h) to (x1, y1, x2, y2) format
    adjusted_pred_boxes_xyxy = adjusted_pred_boxes.copy()
    adjusted_pred_boxes_xyxy[:, 2] = adjusted_pred_boxes_xyxy[:, 0] + adjusted_pred_boxes_xyxy[:, 2]  # x2 = x + w
    adjusted_pred_boxes_xyxy[:, 3] = adjusted_pred_boxes_xyxy[:, 1] + adjusted_pred_boxes_xyxy[:, 3]  # y2 = y + h

    # Apply NMS if nms_threshold > 0
    if nms_threshold >= 0.0:
        # Convert adjusted_pred_boxes and pred_scores to tensors
        boxes_tensor = torch.tensor(adjusted_pred_boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
        labels_tensor = torch.tensor(pred_labels, dtype=torch.int64)

        keep_indices = []

        # Apply NMS per class
        unique_labels = torch.unique(labels_tensor)
        for label in unique_labels:
            label_mask = labels_tensor == label
            boxes_label = boxes_tensor[label_mask]
            scores_label = scores_tensor[label_mask]
            indices = torch.where(label_mask)[0]

            if boxes_label.shape[0] > 0:
                keep = ops.nms(boxes_label, scores_label, nms_threshold)
                keep_indices.extend(indices[keep].tolist())

        # Filter adjusted_pred_boxes, pred_scores, pred_labels
        adjusted_pred_boxes = adjusted_pred_boxes[np.array(keep_indices)]
        adjusted_pred_boxes_xyxy = adjusted_pred_boxes_xyxy[np.array(keep_indices)]
        pred_scores = pred_scores[np.array(keep_indices)]
        pred_labels = pred_labels[np.array(keep_indices)]

    height, width, _ = resized_image.shape
    fig_height = height / 80
    fig_width = width / 80

    # Create subplots: one for the image, one for the precision-recall curve
    fig, (ax_image, ax_pr_curve) = plt.subplots(
        1, 2, figsize=(fig_width * 1.2, fig_height),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
    )
    fig.patch.set_facecolor("white")
    ax_image.imshow(resized_image)
    ax_image.set_xlim([0, width])
    ax_image.set_ylim([height, 0])

    # Ground truth boxes
    for x, y, w, h in adjusted_gt_boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='cyan', linewidth=2)
        ax_image.add_patch(rect)

    # Filter predictions based on confidence threshold
    filtered_indices = pred_scores >= conf_threshold
    filtered_boxes = adjusted_pred_boxes[filtered_indices]
    filtered_boxes_xyxy = adjusted_pred_boxes_xyxy[filtered_indices]
    filtered_scores = pred_scores[filtered_indices]
    filtered_labels = pred_labels[filtered_indices]

    true_positive, false_positive, matched_gt_indices = 0, 0, set()

    # Predicted boxes
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        matching_indices = np.where(gt_labels == label)[0]
        matching_gt_boxes = adjusted_gt_boxes[matching_indices]
        max_iou = 0
        max_iou_idx = -1
        matched_gt_idx = -1
        if len(matching_gt_boxes) > 0:
            iou_values = calculate_iou(box, matching_gt_boxes)
            max_iou_idx = np.argmax(iou_values)
            max_iou = iou_values[max_iou_idx]
            matched_gt_idx = matching_indices[max_iou_idx]

            if max_iou >= detection_iou and matched_gt_idx not in matched_gt_indices:
                color = 'blue'
                true_positive += 1
                matched_gt_indices.add(matched_gt_idx)
            else:
                color = 'purple'
                false_positive += 1
        else:
            color = 'purple'
            false_positive += 1

        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
        ax_image.add_patch(rect)

        # Prepare text properties based on the new option
        text_color = 'white' if scores_with_background else color
        bbox_props = None
        if scores_with_background:
            bbox_props = dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.7)

        ax_image.text(
            x, y - 4, f'{score:.2f}, IoU: {max_iou:.2f}',
            color=text_color,
            fontsize=box_font_size,
            horizontalalignment='left',
            bbox=bbox_props
        )

    false_negative = len(adjusted_gt_boxes) - len(matched_gt_indices)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Display precision and recall with adjustable fonts
    ax_image.text(0.00, 0.99, f'TP: {true_positive:<3} FN: {false_negative:<3} Recall: {recall:.2f}',
                  verticalalignment='bottom', horizontalalignment='left', transform=ax_image.transAxes, color='black', fontsize=stats_font_size)
    ax_image.text(0.00, 0.94, f'FP: {false_positive:<3} TN: 0',
                  verticalalignment='bottom', horizontalalignment='left', transform=ax_image.transAxes, color='black', fontsize=stats_font_size)
    ax_image.text(0.00, 0.89, f'Precision: {precision:.2f}',
                  verticalalignment='bottom', horizontalalignment='left', transform=ax_image.transAxes, color='black', fontsize=stats_font_size)

    # Calculate precision and recall at various confidence thresholds for the current IoU threshold
    confidence_thresholds = np.arange(0.0, 1.01, 0.05)

    # For mAP calculation, use the adjusted predictions after NMS
    mAP_current_iou, original_precisions, original_recalls, adjusted_precisions, recalls, extended_precisions, extended_recalls = compute_precisions_recalls_mAP(
        adjusted_pred_boxes, pred_scores, pred_labels, adjusted_gt_boxes, gt_labels, confidence_thresholds, detection_iou)

    # Plot adjusted PR curve as a thicker line
    ax_pr_curve.plot(extended_recalls, extended_precisions, marker='.', linewidth=2, markersize=6, label='Adjusted', color='blue')

    # Plot original PR curve as a thinner line
    ax_pr_curve.plot(original_recalls, original_precisions, marker='.', linewidth=1.5, markersize=4, label='Original', color='grey')

    # Shade the area under the adjusted PR curve
    ax_pr_curve.fill_between(extended_recalls, extended_precisions, alpha=0.4, color='yellow')

    # Compute mAP@50:95
    detection_ious = np.arange(0.5, 0.96, 0.05)
    mAPs = []
    for iou_thr in detection_ious:
        mAP_iou = compute_mAP(adjusted_pred_boxes, pred_scores, pred_labels,
                              adjusted_gt_boxes, gt_labels, confidence_thresholds, iou_thr)
        mAPs.append(mAP_iou)
    mAP_50_95 = np.mean(mAPs)

    # Change the title to include mAP@current_iou and mAP@50:95
    ax_pr_curve.set_title(f'mAP@{int(detection_iou*100)}: {mAP_current_iou:.2f}, mAP@50:95: {mAP_50_95:.2f}', fontsize=pr_curve_font_size + 2)

    # Make the aspect ratio square
    ax_pr_curve.set_aspect('equal', adjustable='box')

    # Set axis labels
    ax_pr_curve.set_xlabel('Recall', fontsize=pr_curve_font_size)
    ax_pr_curve.set_ylabel('Precision', fontsize=pr_curve_font_size)

    # Set axis limits to always be 0 to 1.05
    ax_pr_curve.set_xlim([0.0, 1.05])
    ax_pr_curve.set_ylim([0.0, 1.05])

    ax_pr_curve.grid(True)

    # Determine legend location based on mAP value
    if mAP_current_iou > 0.5:
        legend_loc = 'lower left'
    else:
        legend_loc = 'upper right'

    # Add a legend
    ax_pr_curve.legend(loc=legend_loc, fontsize=pr_curve_font_size)

    # Plot the orange dot using the precision and recall from the image stats
    current_precision = precision
    current_recall = recall
    ax_pr_curve.plot(current_recall, current_precision, marker='o', color='orange', markersize=10)

    ax_image.axis('off')
    # Use constrained_layout instead of tight_layout to avoid warning
    fig.set_constrained_layout(True)
    plt.show()

def load_annotations(json_file):
    """
    Load annotations from a JSON file and return values as numpy arrays.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    gt_boxes = np.array(data["gt_boxes"])
    gt_labels = np.array(data["gt_labels"])
    pred_boxes = np.array(data["pred_boxes"])
    pred_labels = np.array(data["pred_labels"])
    pred_scores = np.array(data["pred_scores"])

    return gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores




def display_iou_illustration(index, dataset, figsize=(6, 6), alpha=0.5):
    """
    Display an image with ground truth and perturbed masks, illustrating IoU, Dice, Recall, Precision, and Accuracy.

    Parameters:
    - index (int): The index of the sample in the dataset.
    - dataset: The PennFudan dataset.
    - perturbation_fn (callable, optional): A function to perturb the ground truth mask to create a "predicted" mask.
    - figsize (tuple, optional): The size of the display figure (default: (8, 8)).
    """
    def perturbation_fn_factory(quality):
        def perturbation_fn(mask):
            # Create a binary version of the mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Significant dilation
            kernel_size = int(12*(1-quality))  # Fixed large dilation kernel
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            else:
                dilated_mask = binary_mask
            
            # Add a shift to the dilated mask
            shift = -int(10*(1-quality))  # Fixed large shift
            shifted_mask = np.roll(dilated_mask, shift=shift, axis=0)  # Vertical shift
            shifted_mask = np.roll(shifted_mask, shift=shift, axis=1)  # Horizontal shift
            
            # Linearly interpolate between the ground truth and shifted+dilated mask
            interpolated_mask = shifted_mask
            
            # Re-binarize the interpolated mask using a threshold
            interpolated_mask = (interpolated_mask > 0.5).astype(mask.dtype)
            
            return interpolated_mask
        return perturbation_fn

    def update_display(quality):
        # Load the image and ground truth mask
        image, gt_mask = dataset[index]
        gt_mask = gt_mask.numpy().squeeze()

        # Create a perturbed mask as the "predicted" mask
        pred_mask = perturbation_fn_factory(quality)(gt_mask)

        # Calculate TP, FP, FN, and TN
        TP = np.sum((gt_mask > 0.5) & (pred_mask > 0.5))  # True Positives
        FP = np.sum((gt_mask <= 0.5) & (pred_mask > 0.5))  # False Positives
        FN = np.sum((gt_mask > 0.5) & (pred_mask <= 0.5))  # False Negatives
        TN = np.sum((gt_mask <= 0.5) & (pred_mask <= 0.5))  # True Negatives

        # Metrics calculation
        iou = TP / (TP + FP + FN + 1e-6)  # Intersection over Union
        recall = TP / (TP + FN + 1e-6)  # True Positive Rate
        precision = TP / (TP + FP + 1e-6)  # Precision
        accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-6)  # Pixel-wise accuracy
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)  # F1 score for segmentation

        # Debugging: Print metrics breakdown
        #print(f"Metrics Breakdown: TP={intersection}, FN={gt_area - intersection}, FP={pred_area - intersection}")
        #print(f"Precision={precision:.4f}, Recall={recall:.4f}, Dice={dice:.4f}, IoU={iou:.4f}")

        # Convert image to numpy and denormalize
        img_np = image.permute(1, 2, 0).numpy()
        img_np = denormalize_image(img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_np = (img_np * 255).astype(np.uint8)  # Scale for OpenCV display
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Overlay masks with transparency
        overlay = img_bgr.copy()
        overlay = overlay_masks(overlay, (pred_mask > 0.5) & ~(gt_mask > 0.5), color=(0, 165, 255), alpha=alpha)  # orange for false positives
        overlay = overlay_masks(overlay, (gt_mask > 0.5) & ~(pred_mask > 0.5), color=(255, 255, 0), alpha=alpha)  # cyan for false negatives
        overlay = overlay_masks(overlay, (gt_mask > 0.5) & (pred_mask > 0.5), color=(162, 162, 128), alpha=alpha)  # gray for true positives

        # Convert back to RGB for display
        img_rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Display the image with overlaid masks
        plt.figure(figsize=figsize)
        plt.imshow(img_rgb_overlay)
        plt.axis('off')

        # Add metrics text
        plt.text(10, 10, f'IoU: {iou:.2f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(10, 25, f'Dice: {dice:.2f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(10, 40, f'Recall: {recall:.2f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(10, 55, f'Precision: {precision:.2f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(10, 70, f'Accuracy: {accuracy:.2f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Add legend
        legend_elements = [
            Patch(facecolor='gray', edgecolor='gray', label=f'True Positives {TP}'),
            Patch(facecolor='orange', edgecolor='orange', label=f'False Positives {FP}'),
            Patch(facecolor='cyan', edgecolor='cyan', label=f'False Negatives {FN}'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.show()

        # Define a wider layout for the sliders
    slider_layout = widgets.Layout(width='500px')  # Adjust the width as needed
    slider_style = {'description_width': '150px'}  # Adjust the width for descriptions   

    interact(update_display, quality=widgets.FloatSlider(value=0.0, min=0.0, max=1, step=0.01, description='Quality', layout=slider_layout, style=slider_style))
