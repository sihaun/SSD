import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import InvertedResidualBlock

nn.Conv2d(3, 16, 3, 2, 1)

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=160):
      super(MobileNetV3, self).__init__()
      self.model = nn.Sequential(
        InvertedResidualBlock(in_channels=3, expand_channels=16, out_channels=16, kernel_size=3, stride=1, activation="ReLU", se_reduction=1),
        InvertedResidualBlock(16, 64, 24, 3, 2, "ReLU", 1),
        InvertedResidualBlock(24, 72, 24, 3, 1, "ReLU", 1),
        InvertedResidualBlock(24, 72, 40, 5, 2, "ReLU", 16),
        InvertedResidualBlock(40, 120, 40, 5, 1, "ReLU", 16),
        InvertedResidualBlock(40, 120, 40, 5, 1, "ReLU", 16),
        InvertedResidualBlock(40, 240, 80, 3, 2, "Hardswish", 1),
        InvertedResidualBlock(80, 200, 80, 3, 1, "Hardswish", 1),
        InvertedResidualBlock(80, 184, 80, 3, 1, "Hardswish", 1),
        InvertedResidualBlock(80, 184, 80, 3, 1, "Hardswish", 1),
        InvertedResidualBlock(80, 480, 112, 3, 1, "Hardswish", 16),
        InvertedResidualBlock(112, 672, 112, 3, 1, "Hardswish", 16),
        InvertedResidualBlock(112, 672, 160, 5, 2, "Hardswish", 16),
        InvertedResidualBlock(160, 960, 160, 5, 1, "Hardswish", 16),
        InvertedResidualBlock(160, 960, 160, 5, 1, "Hardswish", 16),
      )
      self.fc = nn.Linear(160, num_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
      x = self.model(x)
      x = F.adaptive_avg_pool2d(x, 1)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      return x


def create_mobilenetv3(num_classes : int=1000) -> MobileNetV3:
  return MobileNetV3(num_classes)