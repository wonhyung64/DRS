#%%
import numpy as np
import pandas as pd
from tqdm import tqdm


#%% 설정
# Settings
criteria_seed = 0 # true seed
repeat_num = 1000 # 반복실험 횟수

N = 10000 # number of samples
d = 3 # feature dimension
theta_X_to_T = np.array([2.5, 1.0, -1.5, 0.5]) -50.  # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([2.2, 1.5, -2.0, 0.5, 2.0])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수


#%% DATA GENERATION (Randomized data)
# 1. 독립 변수 X 생성
np.random.seed(criteria_seed)
x = np.random.normal(0, 1, (N, d))  # Normal(0,1)
# x = np.random.uniform(0, 1, (N, d))  # [0,1]  Uniform
design_x = np.concatenate([np.ones([N,1]), x], axis=1)


# 3. 종속 변수 Y 생성 (X와 T의 영향을 받음)
# Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
dummy = np.ones(N).reshape(-1, 1)

# Y(T=1) 생성
X_with_T = np.hstack((design_x, dummy))
logit_p_star = X_with_T.dot(beta_XT_to_Y)
p_star = 1 / (1 + np.exp(-logit_p_star))
np.random.seed(criteria_seed)
y_forward = np.random.binomial(1, p_star)

# Y(T=0) 생성
X_with_T_reverse = np.hstack((design_x, 1 - dummy))
logit_p_star_reverse = X_with_T_reverse.dot(beta_XT_to_Y)
p_star_reverse = 1 / (1 + np.exp(-logit_p_star_reverse))
np.random.seed(criteria_seed)
y_reverse = np.random.binomial(1, p_star_reverse)

true_cate = (1 / (1+np.exp(-beta_XT_to_Y[0] - beta_XT_to_Y[-1]))) - (1 / (1+np.exp(-beta_XT_to_Y[0])))


#%%
random_cate_list = []
real_cate_list = []
ipw_cate_list = []
q_list = []

for repeat_seed in tqdm(range(1, repeat_num+1)):

    ##### randomized data generation #####
    q_star = np.ones(N) * 1/2
    np.random.seed(repeat_seed)
    t_star = np.random.binomial(1, q_star)
    y_star = y_forward * t_star + y_reverse * (1-t_star)

    random_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
    random_df['T'] = t_star
    random_df['Y'] = y_star

    exposed_random_df = random_df[random_df["T"] == 1].reset_index(drop=True)
    y1 = exposed_random_df["Y"].to_numpy()
    unexposed_random_df = random_df[random_df["T"] == 0].reset_index(drop=True)
    y0 = unexposed_random_df["Y"].to_numpy()

    ##### randomized data estimation #####
    random_cate = y1.mean() - y0.mean()
    random_cate_list.append(random_cate)

    ##### real data Y generation #####
    logit_q = design_x.dot(theta_X_to_T)
    q = 1 / (1 + np.exp(-logit_q))
    np.random.seed(repeat_seed)
    t = np.random.binomial(1, q)
    y = y_forward * t + y_reverse * (1-t)

    q_list.append(q)

    real_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
    real_df['T'] = t
    real_df['Y'] = y
    real_df['propensity'] = q

    exposed_real_df = real_df[real_df["T"] == 1].reset_index(drop=True)
    y1 = exposed_real_df["Y"].to_numpy()
    q1 = exposed_real_df["propensity"].to_numpy()

    unexposed_real_df = real_df[real_df["T"] == 0].reset_index(drop=True)
    y0 = unexposed_real_df["Y"].to_numpy()
    q0 = 1 - unexposed_real_df["propensity"].to_numpy()

    real_cate = y1.mean() - y0.mean()
    real_cate_list.append(real_cate)

    ipw_cate = (y1/q1).sum()/N - (y0/q0).sum()/N
    ipw_cate_list.append(ipw_cate)

#%%
print(f"q_bar : {np.mean(q_list).round(4)}\n")
print(f"  True CATE : {round(true_cate,4)}")
print(f"Random CATE : {[(np.array(random_cate_list) - true_cate).mean().round(4), np.array(random_cate_list).var().round(4)]}")
print(f"  Real CATE : {[(np.array(real_cate_list) - true_cate).mean().round(4), np.array(real_cate_list).var().round(4)]}")
print(f"   IPW CATE : {[(np.array(ipw_cate_list) - true_cate).mean().round(4), np.array(ipw_cate_list).var().round(4)]}")
print()

# %%
