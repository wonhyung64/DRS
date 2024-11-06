#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


#%% 설정
criteria_seed = 0
repeat_num = 1000

N = 10000 # number of samples
d = 3 # feature dimension
theta_X_to_T = np.array([2.5, 1.0, -1.5, 0.5])  # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([2.2, 1.5, -2.0, 0.5, 2.0])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수)
num_epochs = 50000


#%% DATA GENERATION (Randomized data)
# 1. 독립 변수 X 생성
np.random.seed(criteria_seed)
x = np.random.normal(0, 1, (N, d))  # 평균 0, 표준편차 1의 정규분포
x = np.concatenate([np.ones([N,1]), x], axis=1)


# 3. 종속 변수 Y 생성 (X와 T의 영향을 받음)
# Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
dummy = np.ones(N).reshape(-1, 1)

##### 3-2. randomized data generation
X_with_T = np.hstack((x, dummy))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p_star = X_with_T.dot(beta_XT_to_Y)
p_star = 1 / (1 + np.exp(-logit_p_star))
np.random.seed(criteria_seed)
y_forward = np.random.binomial(1, p_star)

X_with_T_reverse = np.hstack((x, 1 - dummy))  # X와 T를 결합하여 새로운 독립 변수 행렬 생성
logit_p_star_reverse = X_with_T_reverse.dot(beta_XT_to_Y)
p_star_reverse = 1 / (1 + np.exp(-logit_p_star_reverse))
np.random.seed(criteria_seed)
y_reverse = np.random.binomial(1, p_star_reverse)


#%% True Risk Estimation
xt = X_with_T
y_true = y_forward

np.random.seed(criteria_seed)
true_model = LogisticRegression()
true_model.fit(xt, y_true)
y_pred = true_model.predict_proba(xt)[:,0]
true_risk = log_loss(y_true, y_pred)


#%%
random_risk_list = []
real_risk_list = []
ipw_risk_list = []
random_coef_list = []
real_coef_list = []
ipw_coef_list = []

for repeat_seed in tqdm(range(1, repeat_num+1)):

    ##### 2-2. randomized data generation
    q_star = np.ones(N) * 1/2
    np.random.seed(repeat_seed)
    t_star = np.random.binomial(1, q_star)
    y_star = y_forward * t_star + y_reverse * (1-t_star)

    random_df = pd.DataFrame(x, columns=["intercept"]+[f"X{i+1}" for i in range(d)])
    random_df['T'] = t_star
    random_df['Y'] = y_star

    exposed_random_df = random_df[random_df["T"] == 1].reset_index(drop=True)
    xt = exposed_random_df[["intercept","X1", "X2", "X3", "T"]].to_numpy()
    y_true = exposed_random_df["Y"].to_numpy()

    np.random.seed(repeat_seed)
    random_model = LogisticRegression()
    random_model.fit(xt, y_true, sample_weight=np.ones_like(y_true)*2.)
    y_pred = true_model.predict_proba(xt)[:,0]
    random_risk = log_loss(y_true, y_pred, sample_weight=np.ones_like(y_true)*2.)

    random_coef_list.append(random_model.coef_)
    random_risk_list.append(random_risk)

    #partial
    logit_q = x.dot(theta_X_to_T)
    q = 1 / (1 + np.exp(-logit_q))
    np.random.seed(repeat_seed)
    t = np.random.binomial(1, q)
    y = y_forward * t + y_reverse * (1-t)

    real_df = pd.DataFrame(x, columns=["intercept"]+[f"X{i+1}" for i in range(d)])
    real_df['T'] = t
    real_df['Y'] = y
    real_df['propensity'] = q

    exposed_real_df = real_df[real_df["T"] == 1].reset_index(drop=True)
    xt = exposed_real_df[["intercept", "X1", "X2", "X3", "T"]].to_numpy()
    y_true = exposed_real_df["Y"].to_numpy()
    exposed_q = exposed_real_df["propensity"].to_numpy()

    np.random.seed(repeat_seed)
    real_model = LogisticRegression()
    real_model.fit(xt, y_true)
    y_pred = real_model.predict_proba(xt)[:,0]
    real_risk = log_loss(y_true, y_pred)

    real_coef_list.append(real_model.coef_)
    real_risk_list.append(real_risk)


    np.random.seed(repeat_seed)
    ipw_model = LogisticRegression()
    ipw_model.fit(xt, y_true, sample_weight=1/(2*exposed_q))
    y_pred = ipw_model.predict_proba(xt)[:,0]
    ipw_risk = log_loss(y_true, y_pred, sample_weight=1/(2*exposed_q))

    ipw_coef_list.append(ipw_model.coef_)
    ipw_risk_list.append(ipw_risk)

random_coef_arr = np.concatenate(random_coef_list, 0)
real_coef_arr = np.concatenate(real_coef_list, 0)
ipw_coef_arr = np.concatenate(ipw_coef_list, 0)


#%%
print(f"True Risk : {round(true_risk,4)}")
print(f"Radnom Risk : {np.array(random_risk_list).mean().round(4)} ± {np.array(random_risk_list).std().round(4)}")
print(f"Real Risk : {np.array(real_risk_list).mean().round(4)} ± {np.array(real_risk_list).std().round(4)}")
print(f"IPW  Risk : {np.array(ipw_risk_list).mean().round(4)} ± {np.array(real_risk_list).std().round(4)}")
print()

print(f"True Coef : \n{list(true_model.coef_[0].round(4))}\n")
print(f"Random Coef : \n{[random_coef_arr[:,i].mean().round(4) for i in range(d+2)]} mean")
print(f"{[random_coef_arr[:,i].std().round(4) for i in range(d+2)]} std\n")
print(f"Real Coef : \n{[real_coef_arr[:,i].mean().round(4) for i in range(d+2)]} mean")
print(f"{[real_coef_arr[:,i].std().round(4) for i in range(d+2)]} std\n")
print(f"IPW Coef : \n{[ipw_coef_arr[:,i].mean().round(4) for i in range(d+2)]} mean")
print(f"{[ipw_coef_arr[:,i].std().round(4) for i in range(d+2)]} std\n")
print()

#%%
print(f"True Risk : {round(true_risk,4)}")
print(f"Radnom Risk : {(true_risk - np.array(random_risk_list)).mean().round(4)} ± {(true_risk - np.array(random_risk_list)).std().round(4)}")
print(f"Real Risk : {(true_risk - np.array(real_risk_list)).mean().round(4)} ± {(true_risk - np.array(real_risk_list)).std().round(4)}")
print(f"IPW  Risk : {(true_risk - np.array(ipw_risk_list)).mean().round(4)} ± {(true_risk - np.array(real_risk_list)).std().round(4)}")
print()

print(f"True Coef : \n{list(true_model.coef_[0].round(4))}\n")
print(f"Random Coef : \n{[(true_model.coef_[0,i] - random_coef_arr[:,i]).mean().round(4) for i in range(d+2)]} mean")
print(f"{[(true_model.coef_[0,i] - random_coef_arr[:,i]).std().round(4) for i in range(d+2)]} std\n")
print(f"Real Coef : \n{[(true_model.coef_[0,i] - real_coef_arr[:,i]).mean().round(4) for i in range(d+2)]} mean")
print(f"{[(true_model.coef_[0,i] - real_coef_arr[:,i]).std().round(4) for i in range(d+2)]} std\n")
print(f"IPW Coef : \n{[(true_model.coef_[0,i] - ipw_coef_arr[:,i]).mean().round(4) for i in range(d+2)]} mean")
print(f"{[(true_model.coef_[0,i] - ipw_coef_arr[:,i]).std().round(4) for i in range(d+2)]} std\n")
print()
