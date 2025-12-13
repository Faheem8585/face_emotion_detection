from src.models.baseline import EmotionResNet


def get_model(model_name, num_classes=7, pretrained=True):
    """
    Factory function to get the model.
    Args:
        model_name (str): 'resnet18' or 'mobilenet'
        num_classes (int): Number of emotion classes
        pretrained (bool): Whether to use pretrained weights
    """
    if model_name == 'resnet18':
        return EmotionResNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
