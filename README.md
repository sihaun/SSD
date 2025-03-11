# SSD

### Single Shot MultiBox Detector   
Predicts bounding box coordinates directly without using box generator.
By directly predicting bounding boxes, accurate box coordinates can be obtained.
```Python
        for reg_layer, conf_layer in zip(self.regression_headers, self.classification_headers):
            locs = reg_layer(locs)  # (B, 4, H, W)
            confs = conf_layer(confs)  # (B, num_classes, H, W)
```

## 📌 Key Components

### 1. **PostProcessor**
The `PostProcessor` class handles post-processing for the bounding boxes and scores predicted by the model.

- **Inputs**:  
  - `locs`: List of bounding box coordinates  
  - `confs`: List of class confidence scores  

- **Outputs**:  
  - `result_boxes`: Final selected bounding boxes  
  - `result_labels`: Class labels of the bounding boxes  
  - `result_scores`: Confidence scores of the bounding boxes  
  - `result_confs`: Raw confidence values  

- **Functionality**:  
  - Applies softmax to select the highest-scoring class  
  - Removes low-score predictions based on `score_threshold`  
  - Uses Non-Maximum Suppression (NMS) to eliminate redundant boxes  

---

### 2. **SSD Model**
The `SSD` class consists of the following major components:

- **Backbone**: CNN network for feature map extraction  
- **Extras**: Additional layers to detect objects of different sizes  
- **Classification Headers**: Predict class scores  
- **Regression Headers**: Predict bounding box coordinates  
- **PostProcessor**: Filters and processes detection results  

- **Inputs**:  
  - `x`: Input image tensor  

- **Outputs**:  
  - `result_boxes`: Bounding box coordinates  
  - `result_labels`: Class labels  
  - `result_scores`: Scores of detected objects  
  - `result_confs`: Raw confidence values  

---

### 3. **MultiBoxLoss**
The loss function used for training the SSD model.

- **Loss Function Components**:  
  - `SmoothL1Loss`: Bounding box regression loss  
  - `CrossEntropyLoss`: Classification loss  

- **Inputs**:  
  - `result_boxes`: Predicted bounding boxes  
  - `result_confs`: Predicted class confidence scores  
  - `target_locs`: Ground truth bounding boxes  
  - `target_labels`: Ground truth class labels  

- **Outputs**:  
  - `total_loss`: Total loss (regression loss + classification loss)  
  - `train_acc`: Training accuracy  

- **Hungarian Matching for Alignment**:  
  - Matches predicted and ground truth bounding boxes before computing loss  

---

## ⚙️ How to Run

```bash
python train_ssd.py --arch mobilenetv3_ssd --datapath dataset_coco --epochs 10 --batch-size 5 --lr 0.001 --save ssd.pt
```
