import torch
import numpy as np

def mixup_data(images, labels, alpha=0.2):
    """Mixup: mezcla pares de imágenes y sus labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = images.size()[0]
    index = torch.randperm(batch_size).to(images.device)
    
    mixed_x = lam * images + (1 - lam) * images[index, :]
    y_a, y_b = labels, labels[index]
    return mixed_x, y_a, y_b, lam