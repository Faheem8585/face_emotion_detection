import argparse
import yaml
import torch
import ssl
# Fix for SSL certificate error on macOS when downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.training.trainer import Trainer

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and config['training']['device'] == 'mps':
        device = torch.device('mps')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        config['data']['csv_file'], 
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Model
    model = get_model(
        config['model']['name'], 
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Trainer
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        device, 
        config['training']
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Detection Model")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
