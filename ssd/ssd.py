import torch
import torch.nn as nn
from typing import List, Tuple
import torchvision.ops as ops
from utils.utils import hungarian_matching

class PostProcessor:
    def __init__(
            self, 
            iou_threshold : float=0.5, 
            score_threshold : float=0.5):
        
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def __call__(
            self, 
            locs : List[torch.Tensor], 
            confs : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        result_boxes, result_labels, result_scores, result_confs = [], [], [], []

        final_boxes = torch.empty((0, 4), dtype=torch.float32, device='cuda:0')
        final_labels = torch.tensor([], dtype=torch.int64) 
        final_scores = torch.tensor([], dtype=torch.float32)
        final_confs = torch.tensor([], dtype=torch.float32)

        for loc, conf in zip(locs, confs):
            scores, labels = conf.softmax(dim=-1).max(dim=-1)
            mask = scores > self.score_threshold
            boxes = loc[mask]   
            scores = scores[mask]  
            labels = labels[mask]
            confs = conf[mask]
            
            if boxes.shape[0] == 0: # no boxes
                result_boxes.append((torch.tensor([-1,-1,-1,-1], dtype=torch.float32)))
                result_labels.append(torch.tensor([0], dtype=torch.int64))
                result_scores.append(torch.tensor([-1], dtype=torch.float32))
                result_confs.append(torch.tensor([-1] * len(conf[1]), dtype=torch.float32))
                continue
            
            keep = ops.nms(boxes, scores, self.iou_threshold)
            result_boxes.append(boxes[keep])
            result_labels.append(labels[keep])
            result_scores.append(scores[keep])
            result_confs.append(conf[keep])

            for b, l, s, c in zip(result_boxes, result_labels, result_scores, result_confs):
                final_boxes = b
                final_labels = l
                final_scores = s
                final_confs = c

        return final_boxes, final_labels, final_scores, final_confs
    

class SSD(nn.Module):
    def __init__(
            self, 
            num_classes: int, 
            backbone: nn.ModuleList, 
            source_layer_indexes: List[int],
            post_processor: PostProcessor,
            extras : nn.Sequential, 
            classification_headers : nn.Sequential,
            regression_headers : nn.Sequential, 
            is_test : bool=False, 
            config=None, 
            device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.source_layer_indexes = source_layer_indexes
        self.post_processor = post_processor
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            

    def forward(
            self, 
            x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature Map 추출
        x = self.backbone(x)
        for layer in self.extras:
            x = layer(x)

        locs = confs = x
        # Bounding Box 예측 (cx, cy, w, h)
        for reg_layer, conf_layer in zip(self.regression_headers, self.classification_headers):
            locs = reg_layer(locs)  # (B, 4, H, W)
            confs = conf_layer(confs)  # (B, num_classes, H, W)
        
        # (H, W) 차원을 펼쳐서 (B, N, 4) 형태로 변환
        B, _, H, W = locs.shape
        locs = locs.permute(0, 2, 3, 1).reshape(B, -1, 4)
        confs = confs.permute(0, 2, 3, 1).reshape(B, -1, confs.shape[1])
        
        result_boxes, result_labels, result_scores, result_confs = self.post_processor(locs, confs)

        return result_boxes, result_labels, result_scores, result_confs


    # Loss 함수
class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        self.crossentrophyloss = nn.CrossEntropyLoss()
    
    def forward(
            self, 
            result_boxes : torch.Tensor, 
            result_confs : torch.Tensor, 
            target_locs : torch.Tensor, 
            target_labels : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        reg_loss = torch.tensor(0.0, dtype=torch.float32, device=result_boxes.device, requires_grad=True)
        cls_loss = torch.tensor(0.0, dtype=torch.float32, device=result_boxes.device, requires_grad=True)
        train_acc = torch.tensor(0, dtype=torch.int64, device=result_boxes.device) 

        total_locs = torch.tensor([], dtype=torch.float32)
        total_labels = torch.tensor([], dtype=torch.int64)

        for loc, label in zip(target_locs, target_labels):
            total_locs = torch.cat((total_locs.to(loc.device), loc), dim=0)
            total_labels = torch.cat((total_labels.to(label.device), label), dim=0)
        
        iou_matrix = ops.box_iou(result_boxes, total_locs)

        matched_pred, matched_target = hungarian_matching(iou_matrix)
        for pred, target in zip(matched_pred, matched_target):
            reg_loss = reg_loss + self.l1_loss(result_boxes[pred], total_locs[target])
            cls_loss = cls_loss + self.crossentrophyloss(result_confs[pred], total_labels[target])

            if result_confs[pred].argmax() == total_labels[target]:
                train_acc = train_acc + 1

        return (reg_loss + cls_loss), train_acc
    

