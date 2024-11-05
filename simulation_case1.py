#%%
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#%% 설정
criteria_seed = 0
repeat_seed = 1

N = 1000 # number of samples
d = 3 # feature dimension
theta_X_to_T = np.array([2.5, 1.0, -1.5, 0.5])  # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([2.2, 1.5, -2.0, 0.5, 2.0])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수)
num_epochs = 50000


#%% DATA GENERATION (Randomized data)
# 1. 독립 변수 X 생성
np.random.seed(criteria_seed)
x = np.random.normal(0, 1, (N, d))  # 평균 0, 표준편차 1의 정규분포
x = np.concatenate([np.ones([N,1]), x], axis=1)

# 2. Treatment 변수 T 생성 (X에 영향을 받음)
# T는 X의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수

##### 2-2. randomized data generation
q_star = np.ones(N) * 1/2
np.random.seed(criteria_seed)
t_star = np.random.binomial(1, q_star)


# 3. 종속 변수 Y 생성 (X와 T의 영향을 받음)
# Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수

##### 3-2. randomized data generation
X_with_T = np.hstack((x, t_star.reshape(-1, 1)))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p_star = X_with_T.dot(beta_XT_to_Y)
p_star = 1 / (1 + np.exp(-logit_p_star))
np.random.seed(criteria_seed)
y_forward = np.random.binomial(1, p_star)

X_with_T_reverse = np.hstack((x, 1 - t_star.reshape(-1, 1)))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p_star_reverse = X_with_T_reverse.dot(beta_XT_to_Y)
p_star_reverse = 1 / (1 + np.exp(-logit_p_star_reverse))
np.random.seed(criteria_seed)
y_reverse = np.random.binomial(1, p_star_reverse)

y_star = y_forward * t_star + y_reverse * (1-t_star)


# 4. 데이터 프레임으로 결과 정리
random_df = pd.DataFrame(x, columns=["intercept"]+[f"X{i+1}" for i in range(d)])
random_df['T'] = t_star
random_df['Y'] = y_star


#%% True Risk Estimation
exposed_random_df = random_df[random_df["T"] == 1].reset_index(drop=True)
xt = random_df[["intercept","X1", "X2", "X3", "T"]].to_numpy()
y_true = random_df["Y"].to_numpy()

np.random.seed(criteria_seed)
model = LogisticRegression(random_state=criteria_seed)
model.fit(xt, y_true)

y_pred = model.predict_proba(xt)[:,0]
true_risk = np.mean((-y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)))


#%% DATA GENERATION (Realistic data)
# 2. Treatment 변수 T 생성 (X에 영향을 받음)
# T는 X의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수

##### 2-1. realistic data generation
logit_q = x.dot(theta_X_to_T)
q = 1 / (1 + np.exp(-logit_q))
np.random.seed(repeat_seed)
t = np.random.binomial(1, q)


# 3. 종속 변수 Y 생성 (X와 T의 영향을 받음)
# Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수

##### 3-1. realistic data generation
X_with_T = np.hstack((x, t.reshape(-1, 1)))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p = X_with_T.dot(beta_XT_to_Y)
p = 1 / (1 + np.exp(-logit_p))
np.random.seed(repeat_seed)
y_forward = np.random.binomial(1, p)

X_with_T_reverse = np.hstack((x, 1 - t.reshape(-1, 1)))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p_reverse = X_with_T_reverse.dot(beta_XT_to_Y)
p_reverse = 1 / (1 + np.exp(-logit_p_reverse))
np.random.seed(repeat_seed)
y_reverse = np.random.binomial(1, p_reverse)

y = y_forward * t + y_reverse * (1-t)


# 4. 데이터 프레임으로 결과 정리
real_df = pd.DataFrame(x, columns=["intercept"]+[f"X{i+1}" for i in range(d)])
real_df['T'] = t
real_df['Y'] = y


#%% Risk Estimation with Partial Data
exposed_real_df = real_df[real_df["T"] == 1].reset_index(drop=True)
xt = real_df[["intercept", "X1", "X2", "X3", "T"]].to_numpy()
y_true = real_df["Y"].to_numpy()

np.random.seed(repeat_seed)
real_model = LogisticRegression(random_state=repeat_seed)
real_model.fit(xt, y_true)

y_pred = real_model.predict_proba(xt)[:,0]
real_risk = np.mean((-y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)))

np.random.seed(repeat_seed)
ipw_model = LogisticRegression()
ipw_model.fit(xt, y_true, sample_weight=q)

y_pred = ipw_model.predict_proba(xt)[:,0]
ipw_risk = np.mean((-y_true/q * np.log(y_pred) - (1-y_true)/q * np.log(1-y_pred)))

#%%
ipw_model.coef_
model.coef_
real_model.coef_

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
