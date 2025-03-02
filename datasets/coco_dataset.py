import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}


    def __len__(self):
        return len(self.image_ids)

    
    def x1x2wh_to_cxcywh(self, bbox: List[float]) -> List[float]:
        x_min, y_min, w, h = bbox
        cx = x_min + (w / 2)
        cy = y_min + (h / 2)
        return cx, cy, w, h


    def __getitem__(self, index):
        image_id = self.image_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        image_info = self.coco.imgs[image_id]

        # 이미지 로드
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # numpy.ndarray -> PIL.Image로 변환
        image = Image.fromarray(image)

        # 바운딩 박스 및 레이블 추출
        boxes = []
        labels = []
        data_index = []
        for ann in annotations:
            bbox = ann["bbox"]
            cx, cy, w, h = self.x1x2wh_to_cxcywh(bbox)
            boxes.append([cx, cy, w, h])
            labels.append(ann["category_id"])
            data_index.append(ann["image_id"])
            # 여기에 이미지 id 삽입

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # PyTorch Tensor 변환
        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(data_index, dtype=torch.int64)
        }

        return image, target

