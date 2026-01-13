import torch
import logging

BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-4
NUM_CLASSES = 4  # student, blue_collar, white_collar, retired
MODEL_NAME = "resnet50"  # resnet50, mobilenet_v2 or fairface_yolo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)