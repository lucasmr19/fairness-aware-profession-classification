import torch
from torch import nn
import torchvision.models as models
from ultralytics import YOLO

from .constants import NUM_CLASSES, DEVICE

def create_model(name="resnet50", num_classes=NUM_CLASSES, freeze_backbone=True, unfreeze_layers=3):
    if name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        
        # Mejor arquitectura de clasificación
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            # Congelar todo inicialmente
            for param in model.parameters():
                param.requires_grad = False
            
            # Descongelar últimas capas de ResNet (layer4, layer3, etc.)
            if unfreeze_layers >= 1:
                for param in model.layer4.parameters():
                    param.requires_grad = True
            if unfreeze_layers >= 2:
                for param in model.layer3.parameters():
                    param.requires_grad = True
            if unfreeze_layers >= 3:
                for param in model.layer2.parameters():
                    param.requires_grad = True
            
            # Siempre entrenar la cabeza clasificadora
            for param in model.fc.parameters():
                param.requires_grad = True


    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Replace final classification head
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

        if freeze_backbone:
            # Freeze only the early feature extractor layers
            for layer_name, param in model.features.named_parameters():
                # typical MobileNet early layers: features[0]–[4]
                if any(layer_name.startswith(f"{i}") for i in range(5)):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for param in model.classifier.parameters():
                param.requires_grad = True

    elif name == "fairface_yolo":
        model = YOLO("Anzhc/Race-Classification-FairFace-YOLOv8")
        model.model[-1].nc = num_classes
        model.model[-1].names = ["student", "blue_collar", "white_collar", "retired"]
        model.model[-1].reset_parameters()
        # No backbone freezing logic for YOLO

    else:
        raise ValueError("Unsupported model name.")

    return model.to(DEVICE)