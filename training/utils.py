from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from .constants import DEVICE, LOGGER
from .mixup import mixup_data

from itertools import combinations

# Define classes
RACE_CLASSES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
PROFFESION_CLASSES = ["student", "blue_collar", "white_collar", "retired"]
GENDER_CLASSES = ["Male", "Female"]

def calculate_fairness_for_pairs(
    y_true_int, y_pred_int,
    sensitive_values,
    target_classes=list(range(len(PROFFESION_CLASSES))),
    sensitive_groups=PROFFESION_CLASSES,
    min_samples_per_group=1):
    """
    Computes fairness metrics for:
    - fixed prediction targets (target_classes)
    - across sensitive group pairs (sensitive_groups)
    """

    y_true_int = np.asarray(y_true_int)
    y_pred_int = np.asarray(y_pred_int)
    sensitive_values = np.asarray(sensitive_values)

    if target_classes is None:
        target_classes = list(range(len(np.unique(y_true_int))))
    if sensitive_groups is None:
        sensitive_groups = np.unique(sensitive_values)

    fairness_results = {}

    LOGGER.info(f"Fairness evaluation started | targets={len(target_classes)}, sensitive_groups={len(sensitive_groups)}")

    for target_idx in target_classes:
        y_true_binary = (y_true_int == target_idx).astype(int)
        y_pred_binary = (y_pred_int == target_idx).astype(int)

        positives = y_true_binary.sum()
        if positives == 0 or positives == len(y_true_binary):
            LOGGER.debug(f"Target '{target_idx}' skipped: degenerate positives ({positives}/{len(y_true_binary)})")
            continue

        for g1, g2 in combinations(sensitive_groups, 2):
            pair_name = f"target={target_idx} | groups=({g1},{g2})"
            mask = np.isin(sensitive_values, [g1, g2])

            if not np.any(mask):
                LOGGER.debug(f"{pair_name} skipped: no samples")
                continue

            y_t = y_true_binary[mask]
            y_p = y_pred_binary[mask]
            s_f = sensitive_values[mask]

            unique_groups, counts = np.unique(s_f, return_counts=True)
            group_counts = dict(zip(unique_groups, counts))

            if len(unique_groups) < 2 or np.any(counts < min_samples_per_group):
                LOGGER.debug(f"{pair_name} skipped: insufficient groups or samples (counts={group_counts})")
                continue

            try:
                dp = demographic_parity_difference(y_t, y_p, sensitive_features=s_f)
                eo = equal_opportunity_difference(y_t, y_p, sensitive_features=s_f)
            except ValueError as e:
                LOGGER.warning(f"{pair_name} skipped: {e}")
                continue

            fairness_results[(target_idx, g1, g2)] = {
                "demographic_parity": dp,
                "equal_opportunity": eo,
                "n_samples": len(s_f),
                "group_counts": group_counts,
                "positive_rate": y_t.mean()
            }

            LOGGER.info(f"{pair_name} computed | n={len(s_f)}, counts={group_counts}")

    LOGGER.info(f"Fairness completed: {len(fairness_results)} valid cases")
    return fairness_results

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, preds_all, labels_all = 0, [], []

    for images, labels, sensitive_attr_values in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # MIXUP 50% probability
        use_mixup = np.random.rand() < 0.5

        if use_mixup:
            images_mix, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images_mix)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ===== real labels (NO mixup) =====
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    bal_acc = balanced_accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")
    return running_loss / len(loader), acc, bal_acc, f1

def validate(model, loader, criterion, sensitive_attr=None, gradcam_layers=None, device=DEVICE):
    model.eval()
    y_true, y_pred = [], []
    val_loss_total = 0.0
    sensitive_values = []

    for batch in loader:
        images, labels, sensitive_attr_values = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            sensitive_values.extend(sensitive_attr_values)

            # Optional: Grad-CAM for interpretability
            if gradcam_layers is not None:
                for target_layer in gradcam_layers:
                    for i in range(min(3, images.size(0))):  # visualize 3 images max
                        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
                        grayscale_cam = cam(input_tensor=images[i:i+1], target_category=preds[i].item())[0]
                        img_vis = show_cam_on_image(images[i].cpu().permute(1,2,0).numpy(), grayscale_cam)
                        plt.imshow(img_vis)
                        plt.axis('off')
                        plt.show()

    # Performance metrics
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    f1 = f1_score(y_true_arr, y_pred_arr, average="macro")
    acc = accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)

    # Fairness metrics
    fairness = {}
    if sensitive_attr == "profession":
        fairness = calculate_fairness_for_pairs(
            y_true_arr, y_pred_arr, sensitive_values,
            target_classes=list(range(len(PROFFESION_CLASSES))),
            sensitive_groups=PROFFESION_CLASSES
        )
    elif sensitive_attr == "race":
        fairness = calculate_fairness_for_pairs(
            y_true_arr, y_pred_arr, sensitive_values,
            target_classes=list(range(len(PROFFESION_CLASSES))),
            sensitive_groups=RACE_CLASSES
        )
    elif sensitive_attr == "gender":
        fairness = calculate_fairness_for_pairs(
            y_true_arr, y_pred_arr, sensitive_values,
            target_classes=list(range(len(PROFFESION_CLASSES))),
            sensitive_groups=GENDER_CLASSES
        )

    avg_loss = val_loss_total / len(loader.dataset)
    return avg_loss, acc, bal_acc, f1, fairness

class MetricsTracker:
    """
    Track and store training metrics across epochs.
    Improvement metric can be: 'loss', 'acc', 'bal_acc', or 'f1'.
    Default: 'loss' (minimization).
    """

    def __init__(self, metric: str = "loss"):
        assert metric in ["loss", "acc", "bal_acc", "f1"], \
            "metric must be one of: 'loss', 'acc', 'bal_acc', 'f1'"

        self.metric = metric
        self.minimize = metric == "loss"   # whether to minimize or maximize

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_bal_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_bal_acc': [], 'val_f1': []
        }

        # Best value initialization
        self.best_value = float('inf') if self.minimize else -float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def update(self, epoch: int, train_metrics: dict, val_metrics: dict) -> bool:
        """
        Update metrics for the current epoch.
        Returns:
            bool: True if this epoch produced the best model so far.
        """

        # Store training metrics
        for key in ['loss', 'acc', 'bal_acc', 'f1']:
            self.history[f"train_{key}"].append(train_metrics[key])

        # Store validation metrics
        for key in ['loss', 'acc', 'bal_acc', 'f1']:
            self.history[f"val_{key}"].append(val_metrics[key])

        # --- Improvement check ---
        current_value = val_metrics[self.metric]

        if self.minimize:
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return is_best

    def should_stop_early(self, patience: int) -> bool:
        """Return True if training should stop due to lack of improvement."""
        return self.epochs_without_improvement >= patience

    def get_summary(self) -> str:
        direction = "minimized" if self.minimize else "maximized"
        return (
            f"Best Validation {self.metric} ({direction}): "
            f"{self.best_value:.4f} at epoch {self.best_epoch + 1}"
        )


# Optimizer with differential learning rates
def get_optimizer_with_lr_groups(model, base_lr=1e-4):
    """
    Learning rate diferenciado: cabeza clasificadora aprende más rápido
    que las capas convolucionales
    """
    # Separar parámetros del backbone y la cabeza
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': base_lr * 0.1},  # Backbone learns slower
        {'params': head_params, 'lr': base_lr}             # Head learns at normal rate
    ], weight_decay=1e-4)
    
    return optimizer

# Loss
def get_weighted_loss(train_dataset):
    """Calcula pesos de clase basados en frecuencias inversas"""
    labels = [label.item() for _, label, _ in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(DEVICE),
        label_smoothing=0.1  # Additional regularization
    )