import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.transforms import get_train_transforms, get_val_transforms
from PIL import Image

class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013.
    Expected CSV format: emotion, pixels, Usage
    """
    def __init__(self, csv_file, split='Training', transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            split (str): One of 'Training', 'PublicTest', 'PrivateTest'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform
        
        if not os.path.exists(csv_file):
            # If file doesn't exist, we can't load it. 
            # In a real scenario, we might raise an error or create dummy data.
            # For this implementation, we'll raise an error but provide a helpful message.
            raise FileNotFoundError(f"Dataset file not found at {csv_file}. Please download FER2013.")

        self.data = pd.read_csv(csv_file)
        
        # Filter by split
        if split == 'Training':
            self.data = self.data[self.data['Usage'] == 'Training']
        elif split == 'Validation':
            self.data = self.data[self.data['Usage'] == 'PublicTest']
        elif split == 'Test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']
        
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pixels = self.data.iloc[idx]['pixels']
        emotion = self.data.iloc[idx]['emotion']

        # Convert string of pixels to numpy array
        image = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        
        # Convert to RGB if needed, but FER is grayscale. 
        # Some models expect 3 channels, so we can stack.
        # For now, let's keep it 1 channel and handle in model or transform.
        # Actually, let's convert to 3 channels to be compatible with ResNet/MobileNet pretrained.
        image = np.stack((image,)*3, axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(emotion, dtype=torch.long)

def get_dataloaders(csv_file, batch_size=64, num_workers=4):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    train_dataset = FER2013Dataset(csv_file, split='Training', transform=get_train_transforms())
    val_dataset = FER2013Dataset(csv_file, split='Validation', transform=get_val_transforms())
    test_dataset = FER2013Dataset(csv_file, split='Test', transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
