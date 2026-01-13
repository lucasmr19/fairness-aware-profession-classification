# transforms.py
from torchvision import transforms


train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.90, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.3),

    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),

    transforms.ToTensor(),

    transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.2, 1.8)),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])