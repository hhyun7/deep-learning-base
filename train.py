import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset_a import CustomDataset

#1 데이터 셋 & 데이터로더 연결
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

#2 단순한 모델(Linear Layer)
model = nn.Linear(10,1)

#3 학습 도구 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01) 

print(f"--학습 시작 (총 데이터: {len(dataset)}개) --")

#4 traning Loop
for epoch in range(10):
    for batch_idx, (samples, labels) in enumerate(dataloader):
    # 1 예측
        prediction = model(samples)

    #2 오차 계산
        loss = criterion(prediction, labels)

    #3 미분값 초기화 -> 4 역전파 
        optimizer.zero_grad()
        loss.backward()
        if epoch == 0 and batch_idx == 0:
            print(f"\n[Check] 계산된 Gradient 일부:")
            print(model.weight.grad[0][:5]) #가중치의 미분값 앞 5개만 출력
            print("-> 이 값만큼 가중치 이동.\n")
            
    #5 가중치 갱신
        optimizer.step()

    print(f"Epoch {epoch+1}/10 | Loss {loss.item():.4f}")

print("--학습 완료--")