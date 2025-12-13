import torch.nn as nn
from torchvision import models

class EmotionResNet(nn.Module):
    """
    ResNet-18 based model for emotion detection.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(EmotionResNet, self).__init__()
        # Load ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Modify the final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
