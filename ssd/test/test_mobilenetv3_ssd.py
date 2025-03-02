import torch
import sys, os
sys.path.append('')
from ssd.mobilenetv3_ssd import mobilenetv3_ssd
from ssd.ssd import SSD, PostProcessor, SSDFocalLoss

# 모델, 후처리, 손실 함수 정의
ssd_model = mobilenetv3_ssd(num_classes=21)
nms_processor = PostProcessor(iou_threshold=0.5, score_threshold=0.5)
loss_fn = SSDFocalLoss()
for name, param in ssd_model.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape}")
    print(param)
'''
# 예제 입력
x = torch.randn(2, 3, 300, 300)  # (batch, channels, height, width)
locs, confs = ssd_model(x)
final_boxes = nms_processor(locs, confs)

# 결과 출력
for i, (boxes, labels, scores) in enumerate(final_boxes):
    print(f"Image {i}: {boxes.shape[0]} objects detected")
    '''


