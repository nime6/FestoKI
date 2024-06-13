import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.num_features = self.resnet18.fc.in_features
        self.linear  = nn.Linear(1000, 1)
        self.sig = torch.nn.Sigmoid()
        # Replace the fully connected layer with a new one for binary prediction
        #self.transforms = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #with torch.no_grad():
            #x = self.transforms(x)
            x = self.resnet18(x)
            x = self.linear(x)
            x = self.sig(x) 
            return x.to(float)#.squeeze().to(float)
