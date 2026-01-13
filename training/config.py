from typing import List
from dataclasses import dataclass

from .constants import BATCH_SIZE, EPOCHS, LR, NUM_CLASSES, MODEL_NAME


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LR
    num_classes: int = NUM_CLASSES
    model_name: str = MODEL_NAME # Options: resnet50, mobilenet_v2, fairface_yolo
    num_workers: int = 0
    checkpoint_dir: str = "./checkpoints"
    early_stopping_patience: int = 5
    
    # Class names for interpretability
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["student", "blue_collar", "white_collar", "retired"]
        
        # Validate configuration
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Number of epochs must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_classes == len(self.class_names), "Number of classes must match class names"