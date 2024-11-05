#%%
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 입력 차원을 받아 출력 차원은 1로 설정
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


#%% 설정
seed = 0
n_samples = 1000
n_features = 3
beta_X_to_T = np.array([1.0, -1.5, 0.5])  # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([1.5, -2.0, 0.5, 2.0])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수)
num_epochs = 50000


#%% 데이터 생성
# 1. 독립 변수 X 생성
np.random.seed(seed)
X = np.random.normal(0, 1, (n_samples, n_features))  # 평균 0, 표준편차 1의 정규분포

# 2. Treatment 변수 T 생성 (X에 영향을 받음)
# T는 X의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
linear_combination_T = X.dot(beta_X_to_T)
probabilities_T = 1 / (1 + np.exp(-linear_combination_T))
np.random.seed(seed)
T = np.random.binomial(1, probabilities_T)

# 3. 종속 변수 Y 생성 (X와 T의 영향을 받음)
# Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
X_with_T = np.hstack((X, T.reshape(-1, 1)))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
linear_combination_Y = X_with_T.dot(beta_XT_to_Y)
probabilities_Y = 1 / (1 + np.exp(-linear_combination_Y))
np.random.seed(seed)
Y = np.random.binomial(1, probabilities_Y)

# 4. 데이터 프레임으로 결과 정리
data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_features)])
data['T'] = T
data['Y'] = Y


#%% simulation 1 : ERM w FULL RATING
X = data.drop(columns=["T", "Y"])  # 독립 변수 (X와 T 포함)
X = torch.FloatTensor(X.to_numpy())

y = data['Y']                 # 종속 변수
y = torch.FloatTensor(y.to_numpy())

# 모델 초기화 및 학습 설정
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
model1 = LogisticRegressionModel(n_features)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCELoss(reduction="none")  # 이진 교차 엔트로피 손실
optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)  # 경사하강법 (SGD) 사용

model1_risk = []
# 모델 학습
for epoch in range(num_epochs):
    outputs = model1(X).squeeze()
    loss = criterion(outputs, y.float()).mean()

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    model1_risk.append(loss.item())


#%%
data2 = data[data["T"] == 1].reset_index(drop=True)

X = data2.drop(columns=["T", "Y"])  # 독립 변수 (X와 T 포함)
X = torch.FloatTensor(X.to_numpy())

y = data2['Y']                 # 종속 변수
y = torch.FloatTensor(y.to_numpy())

# 모델 초기화 및 학습 설정
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
model2 = LogisticRegressionModel(n_features)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCELoss(reduction="none")  # 이진 교차 엔트로피 손실
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)  # 경사하강법 (SGD) 사용

model2_risk = []
# 모델 학습
for epoch in range(num_epochs):
    # 모델 예측
    outputs = model2(X).squeeze()
    loss = criterion(outputs, y.float()).mean()

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")

    model2_risk.append(loss.item())


#%%
data2 = data[data["T"] == 1].reset_index(drop=True)

X = data2.drop(columns=["T", "Y"])  # 독립 변수 (X와 T 포함)
t = X.dot(beta_X_to_T)
X = torch.FloatTensor(X.to_numpy())
t = torch.FloatTensor(t.to_numpy()).sigmoid()

y = data2['Y']                 # 종속 변수
y = torch.FloatTensor(y.to_numpy())

# 모델 초기화 및 학습 설정
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
model3 = LogisticRegressionModel(n_features)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCELoss(reduction="none")  # 이진 교차 엔트로피 손실
optimizer = torch.optim.SGD(model3.parameters(), lr=0.01)  # 경사하강법 (SGD) 사용

model3_risk = []
# 모델 학습
for epoch in range(num_epochs):
    # 모델 예측
    outputs = model3(X).squeeze()
    loss = ((criterion(outputs, y.float())+1e-9)/t.float()).sum() / n_samples
    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")

    model3_risk.append(loss.item())


# %%


# 그래프 그리기
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(8, 6))  # 그래프 크기 설정
plt.plot(epochs, model1_risk, label="Full Ratings", color="blue")   # 첫 번째 리스트
plt.plot(epochs, model2_risk, label="Partial Ratings", color="green")  # 두 번째 리스트
plt.plot(epochs, model3_risk, label="Weigted Partial Ratings", color="red")    # 세 번째 리스트

# 그래프에 제목과 레이블 추가
plt.title(f"Empirical Risk over Epochs\n(BCE, n={n_samples}, d={n_features})")
plt.xlabel("Epoch")
plt.ylabel("Value")

# 범례 추가
plt.legend()

# 그래프 표시
plt.show()
# %%

for name, param in model3.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")

# %%
