import torch
import torch.nn as nn
import torch.optim as optim

#1 가짜 데이터 생성
x = torch.randn(4, 10)
y = torch.randn(4,1)

#2 단순한 모델(Linear Layer)
model = nn.Linear(10,1)

#3 학습 도구 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

print("--학습 시작--")

#4 traning Loop
for epoch in range(100):
    # 1 예측
    prediction = model(x)

    #2 오차 계산
    loss = criterion(prediction, y)

    #3 미분값 초기화 -> 4 역전파 -> 5 가중치 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%10 == 0:
        print(f"Epoch {epoch+1}/100 | Loss {loss.item():.4f}")

print("--학습 완료--")