import torch
from torchvision import transforms

def get_train_transforms(image_size=48):
    """
    Returns transformations for the training set.
    Includes augmentation to prevent overfitting.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # Normalize for grayscale images (mean, std)
        # FER2013 is grayscale. If using RGB models, we might need to repeat channels.
        # Here we assume 1 channel.
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

def get_val_transforms(image_size=48):
    """
    Returns transformations for the validation/test set.
    No augmentation, just normalization.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
