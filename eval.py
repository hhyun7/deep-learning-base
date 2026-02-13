import torch

def evaluate(model, detaloader):
    #1. 모델을 평가 모드로 전환
    model.eval()
    correct = 0
    total = 0

    #2. 미분 계산 끄기
    with torch.no_grad():
        for samples, labels in detaloader:

            #입력 데이터 모양 맞추기 (28*28 이미지 -> 784 일렬로 펴기)
            samples = samples.view(-1,28*28)
            outputs = model(samples)
            
            #3. 가장 높은 확률을 가진 정답 고르기
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) #전체 개수
            correct += (predicted == labels).sum().item() #맞춘 개수

    #4. 정확도 계산
    accuracy = 100*correct/total
    return accuracy
