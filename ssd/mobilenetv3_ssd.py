import torch
import torch.nn as nn
from torch.nn import Conv2d, ModuleList
from ssd import SSD, PostProcessor
from models import MobileNetV3
from utils import InvertedResidualBlock


def mobilenetv3_ssd(num_classes, is_test=False):
    base_net = MobileNetV3().model
    source_layer_indexes = [
        15,
        16,
    ]

    postProcessor = PostProcessor()
    
    extras = nn.Sequential(
        InvertedResidualBlock(160, 256, 512, kernel_size=3, stride=1, activation='Hardswish'),
        InvertedResidualBlock(512, 128, 256, kernel_size=3, stride=1, activation='Hardswish'),
        InvertedResidualBlock(256, 128, 256, kernel_size=3, stride=1, activation='Hardswish'),
        InvertedResidualBlock(256, 128, 256, kernel_size=3, stride=1, activation='Hardswish'),
    )

    regression_headers = nn.Sequential(
        Conv2d(in_channels=256, out_channels=4, kernel_size=3, padding=1),
    )
    classification_headers = nn.Sequential(
        Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, padding=1),
    )
    for reg, cls in zip(regression_headers, classification_headers):
        if isinstance(reg, nn.Linear):  # Linear 레이어에 대해서만
            reg.weight.data.zero_()  # 가중치 0으로 초기화
            if reg.bias is not None:
                reg.bias.data.zero_() 
        if isinstance(cls, nn.Linear):  # Linear 레이어에 대해서만
            cls.weight.data.zero_()  # 가중치 0으로 초기화
            if cls.bias is not None:
                cls.bias.data.zero_() # 바이어스 0으로 초기화
        
    return SSD(num_classes, base_net, source_layer_indexes, postProcessor,
               extras, classification_headers, regression_headers, is_test=is_test)