import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset_a import CustomDataset


# simple model import
from models.simple_model import SimpleModel

#1 데이터 셋 & 데이터로더 연결
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

#2 모델 준비(수정)
#입력 10개 -> 출력 1개인 모델 생성
model = SimpleModel(input_dim=10, output_dim=1)

#3 학습 도구 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01) #model.parameters() : fc1, fc2의 weight, bias

print(f"--학습 시작--")

#4 traning Loop
for epoch in range(10):
    epoch_loss = 0.0
    for batch_idx, (samples, labels) in enumerate(dataloader):
    # 1 예측
        prediction = model(samples)

    #2 오차 계산
        loss = criterion(prediction, labels)

    #3 미분값 초기화 -> 4 역전파 
        optimizer.zero_grad()
        loss.backward()
            
    #5 가중치 갱신
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/10 | Avg Loss {epoch_loss / len(dataloader):.4f}")

print("--학습 완료--")