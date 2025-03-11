import torch
import warnings
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets.coco_dataset import COCODataset
from tqdm import tqdm
import numpy as np
import sys
import config
import time
import utils
import ssd

warnings.simplefilter('ignore')

def hyperparam():
    args = config.config()
    return args

class Train_SSD():
    def __init__(
            self, 
            data_dir : str, 
            batch_size : int):
        # GPU, 글씨체 설정(Windows OS)
        if os.name == 'nt' and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU can be available")
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:
            raise Exception('No CUDA found')
        # (-) 설정
        plt.rcParams['axes.unicode_minus'] = 'False'

        # instances   
        self.test_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ])
        self.train_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])
        # 폴더 경로를 추가
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = data_dir
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'val')

        # 훈련용
        self.train_data = COCODataset(train_dir, annotation_file=os.path.join(train_dir,'_annotations.coco.json'),
                    transform=self.train_transform)
        # 검증용
        self.test_data = COCODataset(test_dir, annotation_file=os.path.join(test_dir,'_annotations.coco.json'),
                    transform=self.test_transform)    
   
        # 데이터로더 정의
        self.batch_size = batch_size
        # 훈련용
        self.train_loader = DataLoader(self.train_data, 
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 검증용
        self.test_loader = DataLoader(self.test_data, 
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 이미지 출력용
        self.test_loader2 = DataLoader(self.test_data, 
            batch_size=50, shuffle=True, num_workers=4, pin_memory=True)

        # model
        self.arch = None
        self.net = None          


    def prepare_model(
            self, 
            arch : str, 
            label_file_path : str, 
            weight_path : str):
        self.arch = arch
        self.labels = np.loadtxt(label_file_path, str, delimiter='\t')
        num_classes = len(self.labels)
        self.net = ssd.__dict__[self.arch](num_classes)
        self.net = self.net.to(self.device)
        if weight_path != 'default':
            self.net.load_state_dict(torch.load(weight_path, map_location=self.device))


    def fit(self, lr : float=0.001, epochs : int=10) -> np.ndarray:
        self.lr = lr  
        self.optimizer = optim.Adam(self.net.parameters(),lr=lr)
        self.nms_processor = ssd.PostProcessor()
        self.criterion = ssd.MultiBoxLoss()
        self.epoch = epochs
        self.history = np.zeros((0, 5))
        base_epochs = len(self.history)
    
        for epoch in range(base_epochs, self.epoch+base_epochs):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            # 훈련 페이즈
            self.net.train()
            count = 0

            for image, target in tqdm(self.train_loader):
                image = image.to(self.device)
                boxes = target['boxes'].to(self.device)
                labels = target['labels'].to(self.device)
                image_id = target['image_id'].to(self.device) 
                count += len(labels)

                # 경사 초기화
                self.optimizer.zero_grad()

                # 예측 계산
                result_boxes, result_labels, result_scores, result_confs = self.net(image)

                loss, acc = self.criterion(result_boxes, result_confs, boxes, labels)
                train_loss += loss.item()
                train_acc += acc.item()

                loss.backward()

                avg_train_loss = train_loss / count
                avg_train_acc = train_acc / count

                # 파라미터 수정
                self.optimizer.step()

            # 예측 페이즈
            self.net.eval()
            count = 0

            for image, target in self.test_loader:
                image = image.to(self.device)
                boxes = target['boxes'].to(self.device)
                labels = target['labels'].to(self.device)      
                count += len(labels) 

                # 예측 계산

                result_boxes, result_labels, result_scores, result_confs = self.net(image)

                loss, acc = self.criterion(result_boxes, result_confs, boxes, labels)

                val_loss += loss.item()
                val_acc += acc.item()

                avg_val_loss = val_loss / count
                avg_val_acc = val_acc / count
        
            print (f'Epoch [{(epoch+1)}/{self.epoch+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
            item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
            self.history = np.vstack((self.history, item))
        return self.history
    

    def evaluate_history(self):
        # Check Loss and Accuracy
        print(f'Initial state: Loss: {self.history[0,3]:.5f}  Accuracy: {self.history[0,4]:.4f}') 
        print(f'Final state: Loss: {self.history[-1,3]:.5f}  Accuracy: {self.history[-1,4]:.4f}')

        num_epochs = len(self.history)
        unit = num_epochs / 10

        # Plot the training curve (Loss)
        plt.figure(figsize=(9,8))
        plt.plot(self.history[:,0], self.history[:,1], 'b', label='Training')
        plt.plot(self.history[:,0], self.history[:,3], 'k', label='Validation')
        plt.xticks(np.arange(0, num_epochs+1, unit))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Curve (Loss)')
        plt.legend()
        plt.show()

        # Plot the training curve (Accuracy)
        plt.figure(figsize=(9,8))
        plt.plot(self.history[:,0], self.history[:,2], 'b', label='Training')
        plt.plot(self.history[:,0], self.history[:,4], 'k', label='Validation')
        plt.xticks(np.arange(0, num_epochs+1, unit))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Curve (Accuracy)')
        plt.legend()
        plt.show()


    def show_images_labels(self):
        """
        SSD 모델의 예측 결과를 시각화하는 함수
        """
        for images, targets in self.test_loader:
            break  # 첫 번째 배치만 사용
        
        images = images.to(self.device)
        
        # 모델 예측
        with torch.no_grad():
            locations, confidences = self.net(images)
            result_boxes, result_labels, result_scores, _ = self.nms_processor(locations, confidences)
        
        # 플롯 크기 설정
        batch_size = len(images)
        plt.figure(figsize=(20, min(5, batch_size) * 3))
        
        for i in range(batch_size):
            ax = plt.subplot(5, 10, i + 1)  # 최대 5행 10열 표시
            
            # 이미지 변환
            image_np = images[i].cpu().numpy().transpose((1, 2, 0))
            image_np = (image_np + 1) / 2  # 정규화 해제
            plt.imshow(image_np)
            
            # GT 바운딩 박스 표시
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            
            for box, label in zip(gt_boxes, gt_labels):
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, self.classes[label], color='g', fontsize=10, weight='bold')
            
            # 예측 바운딩 박스 표시
            for j in range(len(result_boxes[i])):
                box = result_boxes[i][j].cpu().numpy()
                label = result_labels[i][j].cpu().numpy()
                score = result_scores[i][j].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 10, f'{self.classes[label]} {score:.2f}',
                        color='r', fontsize=10, weight='bold')
            
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = hyperparam()
    t1 = Train_SSD(args.datapath, args.batch_size)
    t1.prepare_model(args.arch,'label_map.txt', args.load_weights)
    print(args)
    utils.torch_seed()
    start_time = time.time()
    t1.fit(args.lr, args.epochs)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))    
    utils.save_weight(net=t1.net, path=args.save)
    t1.evaluate_history()
    utils.torch_seed()
    #t1.show_images_labels()