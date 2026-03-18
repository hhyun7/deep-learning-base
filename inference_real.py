from PIL import Image
import torch
import torchvision.transforms as transforms
from models.cnn_model import CNNModel

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])


img=Image.open('my_digit2.png')
img=transform(img)
img = 1.0-img
img = img.unsqueeze(0)

#1. 평가 모델 만들기(학습 때와 같은 구조)
model = CNNModel(input_dim=784, output_dim=10)

#2. 저장된 가중치 불러오기
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval() #평가 모드

#3 진짜 데이터 1장 꺼내서 예측
#test_dataset=datasets.MNIST(root= './data', train=False, transform=transforms.ToTensor())
image = img # 첫 번째 데이터 꺼내기

with torch.no_grad():
    #input_tensor = image.view(1, 784) 
    #RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 784]
    #cnn -> view 지우고 그대로 사용
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"실제 정답: 7")
print(f"모델 예측: {predicted.item()}")