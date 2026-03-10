import torch
import torch.nn as nn

#nn.Module - 학습 가능한 객체 등록해줌, 모든 딥러닝 모델이 상속 받아야 함
class SimpleModel (nn.Module):
    def __init__(self, input_dim, output_dim):  #__init__ : 연산 안함, 부품 생성만
        super(SimpleModel, self).__init__()

        #입력 -> 은닉층
        self.fc1 = nn.Linear(input_dim, 32)
        
        #은닉층 : 모델 내부에서 특징을 변환하는 공간
        self.relu = nn.ReLU() #ReLU -> 비선형성 추가 : 음수 -> 0 / 양수 -> 그대로

        #은닉층 -> 출력
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x): #forward : 예측 함수 / 데이터 흐름도(x->Wx+b->ReLU->Wx+b->output)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out