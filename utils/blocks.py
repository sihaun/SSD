import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 expand_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 activation : str, 
                 se_reduction=1):
        super(InvertedResidualBlock, self).__init__()
        self.pointwise1 = PointwiseConv2d(in_channels, 
                                          expand_channels, 
                                          activation=activation)
        self.depthwise = DepthwiseConv2d(expand_channels, 
                                         expand_channels, 
                                         kernel_size, 
                                         stride=stride, 
                                         activation=activation)
        self.pointwise2 = PointwiseConv2d(expand_channels, 
                                          out_channels, 
                                          activation=activation)
        self.se = None
        if se_reduction > 1:
            self.se = SEBlock(out_channels, se_reduction)
        elif se_reduction < 1:
            torch._assert(False, f"SE Reduction {se_reduction} must be greater than or equal to 1.")

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.pointwise2(x)
        if self.se is not None:
            x = self.se(x)
        return x
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.conv = Conv2dNormActivation(in_channels, 
                                         out_channels, 
                                         kernel_size=kernel_size, 
                                         stride=stride, 
                                         groups=in_channels, 
                                         **kwargs)

    def forward(self, x):
        return self.conv(x)
    
class PointwiseConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 **kwargs):
        super(PointwiseConv2d, self).__init__()
        self.conv = Conv2dNormActivation(in_channels, 
                                         out_channels, 
                                         kernel_size=1, 
                                         **kwargs)

    def forward(self, x):
        return self.conv(x)


class Conv2dNormActivation(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 kernel_size, 
                 stride=1, 
                 groups=1, 
                 activation='ReLU', 
                 **kwargs):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(input_channels, 
                              output_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=kernel_size//2, 
                              groups=groups, 
                              **kwargs)
        self.norm = nn.BatchNorm2d(output_channels)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.norm.weight.data.fill_(1)
        self.norm.bias.data.zero_()

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Hardswish':
            self.activation = nn.Hardswish()
        else:
            torch._assert(False, f"Activation function {activation} is not supported.")


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Channel reduction을 위한 첫 번째 FC layer
        self.fc1 = nn.Linear(channel, channel // reduction)
        # 복원되는 FC layer
        self.fc2 = nn.Linear(channel // reduction, channel)
        # Sigmoid 활성화 함수 (채널 중요도 계산)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling (GAP)
        b, c, _, _ = x.size()  # b: batch size, c: channels
        gap = torch.mean(x, dim=(2, 3), keepdim=False)  # (B, C) 형태로 GAP을 계산

        # Fully Connected layer를 통해 중요도 학습
        gap = self.fc1(gap)  # (B, C//reduction)
        gap = torch.relu(gap)  # ReLU 활성화
        gap = self.fc2(gap)  # (B, C)

        # Sigmoid를 통해 0과 1 사이의 값으로 중요도 조절
        gap = self.sigmoid(gap).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        # 입력 특성 맵에 채널별 중요도를 곱하여 가중치를 부여
        return x * gap  # (B, C, H, W) 형태로 스케일링된 출력
 