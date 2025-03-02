import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.init as init
from scipy.optimize import linear_sum_assignment
# 모델 가중치 저장
def save_weight(net, path='weight.pt'):
   torch.save(net.state_dict(), path)

def save_model(net, path='model.pt'):
   torch.save(net, path)

# 모델 가중치 불러오기
def load_weights(net, path='weight.pth'):
   net.load_state_dict(torch.load(path))

# 모델 가중치 확인
def show_weights(net):
    for name, param in net.state_dict().items():
        print(f"Layer: {name} | Shape: {param.shape}")
        print(param)


# 파이토치 난수 고정
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

import torchvision.models.detection
'''
padding
kernel=1 => padding=0
kernel=3 => padding=1
kernel=5 => padding=2

stride
stride=1 => upsample
stride=2 => downsample

depthwise conv => (group=in_channels)
nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

pointwise conv => (kerbel_size=1), stride=1, padding=0
nn.Conv2d(in_channels, out_channels, kernel_size=1)

=> Inverted Residual Block
pointwise -> depthwise -> pointwise -> se
input, output, kernel
expand, stride, se, activation

            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
'''
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

        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
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
    

def hungarian_matching(iou_matrix):
    """
    헝가리안 알고리즘을 이용해 최적의 바운딩 박스 매칭 수행
    
    Args:
        iou_matrix (Tensor): IoU 행렬 (N, M)
        
    Returns:
        matched_pred (Tensor): 예측 박스 매칭 인덱스
        matched_target (Tensor): 정답 박스 매칭 인덱스
    """
    # IoU 행렬을 numpy 배열로 변환
    cost_matrix = -iou_matrix.detach().cpu().numpy()  # IoU가 최대가 되어야 하므로 -1을 곱하여 최소화 문제로 변환
    pred_indices, target_indices = linear_sum_assignment(cost_matrix)

    return torch.tensor(pred_indices), torch.tensor(target_indices)