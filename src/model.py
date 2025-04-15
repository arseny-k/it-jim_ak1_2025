import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(model_name="resnet18"):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
    return model
