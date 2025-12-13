import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.epochs = config['epochs']
        self.save_dir = config['save_dir']
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.writer.add_scalar('Val/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
        self.writer.add_scalar('Val/F1', epoch_f1, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {self.train_loss:.4f}, Train Acc: {self.train_acc:.4f}, Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
        
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            print(f"Saved best model with acc: {self.best_val_acc:.4f}")
            
        return epoch_loss, epoch_acc

    def train(self):
        for epoch in range(self.epochs):
            self.train_loss, self.train_acc = self.train_epoch(epoch)
            self.validate(epoch)
            
            if self.scheduler:
                self.scheduler.step(self.train_loss) # Assuming ReduceLROnPlateau
                
        self.writer.close()
