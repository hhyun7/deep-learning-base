import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# simple model import
from models.simple_model import SimpleModel
from eval import evaluate # 채점기

#1 데이터 불러오기
transform = transforms.Compose([transforms.ToTensor()])

#학습용 데이터 60,000장
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#시험용 데이터 10,000장
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#2 모델 준비 (크기 변경)
#입력: 28*28 이미지 = 784 픽셀
#출력: 숫자 0~9 = 10개 클래스
model = SimpleModel(input_dim=784, output_dim=10)

#3 설정 (Loss 변경)
#분류 문제는 CrossEntropyLoss를 씀
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f"--학습시작 (데이터 {len(train_dataset)}장)--")
print(f"학습 전: {evaluate(model, test_loader):.2f}")
#4 training loop
epochs = 5
for epoch in range(epochs):
    
    #학습 모드
    model.train()
    running_loss = 0.0

    for samples, labels in train_loader:
        #이미지 펼치기
        samples = samples.view(-1, 28*28)

        #예측 -> 오차
        outputs = model(samples)
        loss = criterion(outputs, labels)
        
        #역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    #Epoch 끝날 때마다 시험보기
    train_acc = evaluate(model, train_loader) #공부한 걸로 시험(보통 점수 높음)
    test_acc = evaluate(model, test_loader) #안 본 걸로 시험(진짜 실력)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f} | Test Acc: {train_acc:.2f}%")

print("--학습 완료--")