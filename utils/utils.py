import torch
from typing import List, Tuple
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
   

def hungarian_matching(iou_matrix : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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