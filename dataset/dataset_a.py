import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.randn(1000, 10)
        #self.y_data = torch.randn(1000,1) 데이터 양이 100개 인데 규칙이 없고 랜덤이어서 패턴을 찾을 수 없음
        #이전에는 데이터 양이 4개여서 Overfitting 상태 - 암기해서 다 맞춰버림 새로운 데이터 들어오면 틀림
        #위 사황에서는 규칙이 필요 없음
        
        self.y_data = torch.sum(self.x_data, dim = 1, keepdim=True) 

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x,y