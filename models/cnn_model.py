import torch
import torch.nn as nn

#nn.Module - 학습 가능한 객체 등록해줌, 모든 딥러닝 모델이 상속 받아야 함
class CNNModel (nn.Module):
    def __init__(self, input_dim, output_dim):  #__init__ : 연산 안함, 부품 생성만
        super(CNNModel, self).__init__()

        self.fc1 = nn.Conv2d(1,16, 3)
        self.relu1 = nn.ReLU() #ReLU -> 비선형성 추가 : 음수 -> 0 / 양수 -> 그대로
        self.fc2 = nn.MaxPool2d(2)

        self.fc3 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.30) #Dropout: Overfitting 방지 위해 무작위로 신경망의 일부 차단
        #test1 0.25 / epoch 8 : test acc 97.59
        #test2 0.30 / epoch 8 : test acc 97.66
        #test3 0.35 / epoch 10 : test acc 97.86
        #test4 0.5 / epoch 10 : test acc 97.90
        #test5 0.5 / epoch 15 : test acc 98.00

        self.fc5 = nn.Linear(800,output_dim)

    def forward(self, x): #forward : 예측 함수 / 데이터 흐름도(x->Wx+b->ReLU->Wx+b->output)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = self.fc4(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc5(out)
        
        return out