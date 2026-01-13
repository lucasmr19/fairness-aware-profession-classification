import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Dataset as HFDataset
from typing import Tuple
import os

from .constants import LOGGER
from .model_creation import create_model
from .config import TrainingConfig
from .utils import MetricsTracker, train_one_epoch, validate, get_optimizer_with_lr_groups, get_weighted_loss
from .dataset import FairFaceDataset
from .transforms import train_transforms, valid_transforms

def train_model(
    train_ds: HFDataset,
    valid_ds: HFDataset,
    config: TrainingConfig,
    experiment_name: str,
    sensitive_attr: str = "profession"
) -> Tuple[nn.Module, MetricsTracker]:
    """
    Main training for a classification model with fine-tuning.
    
    Args:
        train_df: Training Dataset with 'profession' columns
        valid_df: Validation Dataset with 'profession' columns
        config: Training configuration
        experiment_name: Name of the experiment for logging and checkpointing
    
    Returns:
        Tuple of (trained_model, metrics_tracker)
    """
    LOGGER.info(f"{'='*70}")
    LOGGER.info(f"Starting training: Experiment '{experiment_name}' | Model: {config.model_name}")
    LOGGER.info(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # 1. Prepare data loaders
    # -------------------------------------------------------------------------
    LOGGER.info("Preparing datasets...")
    train_dataset = FairFaceDataset(train_ds, transform=train_transforms, sensitive_attr=sensitive_attr)
    valid_dataset = FairFaceDataset(valid_ds, transform=valid_transforms, sensitive_attr=sensitive_attr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    LOGGER.info(f"Training samples: {len(train_dataset)} | Validation samples: {len(valid_dataset)}")
    
    # -------------------------------------------------------------------------
    # 2. Initialize model, loss, optimizer
    # -------------------------------------------------------------------------
    LOGGER.info("Initializing model...")
    model = create_model(config.model_name, num_classes=config.num_classes)
    criterion = get_weighted_loss(train_dataset)
    optimizer = get_optimizer_with_lr_groups(model, base_lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-4],
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # -------------------------------------------------------------------------
    # 3. Resume from checkpoint if exists
    # -------------------------------------------------------------------------
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{experiment_name}_{config.model_name}_best.pth"
    )

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        LOGGER.info(f" Resuming training from epoch {start_epoch}")
    else:
        LOGGER.info(" Starting training from scratch")
    
    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    metrics_tracker = MetricsTracker()
    
    LOGGER.info("Starting training loop...")
    for epoch in range(start_epoch, start_epoch + config.epochs):
        LOGGER.info(f"\n--- Epoch [{epoch+1}/{start_epoch + config.epochs}] ---")
        
        # Train
        train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        
        # Validate
        val_loss, val_acc, val_bal_acc, val_f1, fairness_metrics = validate(
            model,
            valid_loader,
            criterion,
            sensitive_attr=sensitive_attr,  # adjust as needed
            gradcam_layers=None,    # list of layers for Grad-CAM
            device=next(model.parameters()).device
        )
        
        # Update metrics
        train_metrics = {
            'loss': train_loss, 'acc': train_acc,
            'bal_acc': train_bal_acc, 'f1': train_f1
        }
        val_metrics = {
            'loss': val_loss, 'acc': val_acc,
            'bal_acc': val_bal_acc, 'f1': val_f1
        }
        
        is_best = metrics_tracker.update(epoch, train_metrics, val_metrics)
        
        # Log metrics
        LOGGER.info(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                   f"Bal.Acc: {train_bal_acc:.4f} | F1: {train_f1:.4f}")
        LOGGER.info(f"Valid → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
                   f"Bal.Acc: {val_bal_acc:.4f} | F1: {val_f1:.4f}")
        
        for pair, metrics in fairness_metrics.items():
            print(f"Comparing {pair[0]} vs {pair[1]}")
            print(f"Demographic Parity: {metrics['demographic_parity']:.4f}")
            print(f"Equal Opportunity: {metrics['equal_opportunity']:.4f}")
            print("-----")
        
        # Save best model
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            LOGGER.info(f"Best model saved at path: {checkpoint_path}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if metrics_tracker.should_stop_early(config.early_stopping_patience):
            LOGGER.warning(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # -------------------------------------------------------------------------
    # 5. Training summary
    # -------------------------------------------------------------------------
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info(f"Training completed: {experiment_name}")
    LOGGER.info(metrics_tracker.get_summary())
    LOGGER.info(f"{'='*70}\n")
    
    return model, metrics_tracker    