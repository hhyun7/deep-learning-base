import torch
from torchvision import datasets, transforms
from models.simple_model import SimpleModel

#1. 평가 모델 만들기(학습 때와 같은 구조)
model = SimpleModel(input_dim=784, output_dim=10)

#2. 저장된 가중치 불러오기
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval() #평가 모드

#3 진짜 데이터 1장 꺼내서 예측
test_dataset=datasets.MNIST(root= './data', train=False, transform=transforms.ToTensor())
image, label = test_dataset[0] # 첫 번째 데이터 꺼내기

with torch.no_grad():
    input_tensor = image.view(1, 784) 
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

print(f"실제 정답: {label}")
print(f"모델 예측: {predicted.item()}")