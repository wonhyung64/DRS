#%%
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


#%% 설정
# Settings
repeat_num = 1000 # 반복실험 횟수

N = 10000 # number of samples
# N = 1000000 # number of samples
d = 3 # feature dimension
treatment_effect = 2.0
theta_X_to_T_ = np.array([2.5, 1.0, -1.5, 0.5])   # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([2.2, 1.5, -2.0, 0.5, treatment_effect])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수

x_dist = "normal"
# x_dist = "uniform"

theta_loc_list1 = [0., 2., 5.,] #normal 10^4
theta_loc_list2 = [0.32, 0.8, 1.5] #uniform 10^4
# theta_loc_list = [0., 0.3, 1.5, 3.] #uniform 10^6

#%%
for (x_dist, N ,theta_loc_list) in [
    ("normal", 10000, theta_loc_list1),
    ("uniform", 10000, theta_loc_list2),
    # ("uniform", 1000000, theta_loc_list3),
    ]:

    for theta_loc in theta_loc_list:
        print(f"theta_loc : {theta_loc}")

        theta_X_to_T = theta_X_to_T_ - theta_loc
        beta_XT_to_Y = np.array(list(-theta_X_to_T_) + [treatment_effect])

        random_bias_list = []
        real_bias_list = []
        ipw_bias_list = []
        random_coef_list = []
        real_coef_list = []
        ipw_coef_list = []
        q_list = []

        random_mean_list = []
        real_mean_list = []
        ipw_mean_list = []

        random_ate_list = []
        real_ate_list = []
        ipw_ate_list = []

        cum_p_star = 0.
        cum_p_star_reverse = 0.

        for repeat_seed in tqdm(range(1, repeat_num+1)):
            np.random.seed(repeat_seed)

            # 독립 변수 X 생성
            if x_dist == "normal":
                x = np.random.normal(0, 1, (N, d))  # Normal(0,1)
            elif x_dist == "uniform":
                x = np.random.uniform(0, 1, (N, d))  # [0,1]  Uniform
            design_x = np.concatenate([np.ones([N,1]), x], axis=1)

            # 종속 변수 Y 생성 (X와 T의 영향을 받음)
            # Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
            dummy = np.ones(N).reshape(-1, 1)

            # Y(T=1) 생성
            X_with_T = np.hstack((design_x, dummy))
            logit_p_star = X_with_T.dot(beta_XT_to_Y)
            p_star = 1 / (1 + np.exp(-logit_p_star))
            y_forward = np.random.binomial(1, p_star)

            # Y(T=0) 생성
            X_with_T_reverse = np.hstack((design_x, 1 - dummy))
            logit_p_star_reverse = X_with_T_reverse.dot(beta_XT_to_Y- np.array([4., 0., 0., 0., 0.]))
            p_star_reverse = 1 / (1 + np.exp(-logit_p_star_reverse))
            y_reverse = np.random.binomial(1, p_star_reverse)

            cum_p_star += p_star.sum()
            cum_p_star_reverse += p_star_reverse.sum()

            ##### randomized data generation #####
            q_star = np.ones(N) * 1/2
            t_star = np.random.binomial(1, q_star)
            y_star = y_forward * t_star + y_reverse * (1-t_star)

            random_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
            random_df['T'] = t_star
            random_df["Y"] = y_star

            ##### ranomized mean estimation #####
            exposed_random_df = random_df[random_df["T"] == 1].reset_index(drop=True)
            y1 = exposed_random_df["Y"].to_numpy()
            unexposed_random_df = random_df[random_df["T"] == 0].reset_index(drop=True)
            y0 = unexposed_random_df["Y"].to_numpy()
            random_mean = y1.mean() - y0.mean()
            random_mean_list.append(random_mean)

            #### randomized com estimation #####
            random_xt = random_df[["X1", "X2", "X3", "T"]].to_numpy()
            # random_xt = random_df[["T"]].to_numpy()
            random_model = LogisticRegression(C=10000)
            random_model.fit(random_xt, y_star)
            random_bias_list.append(random_model.intercept_)
            random_coef_list.append(random_model.coef_)
            random_ate_list.append(
                (random_model.predict_proba(X_with_T[:,1:])[:,1] - random_model.predict_proba(X_with_T_reverse[:,1:])[:,1]).mean()
            )

            ##### real data Y generation #####
            logit_q = design_x.dot(theta_X_to_T)
            q = 1 / (1 + np.exp(-logit_q))
            t = np.random.binomial(1, q)
            y = y_forward * t + y_reverse * (1-t)

            q_list.append(q.mean())

            real_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
            real_df['T'] = t
            real_df['Y'] = y
            real_df['propensity'] = q

            ##### real mean estimation
            exposed_real_df = real_df[real_df["T"] == 1].reset_index(drop=True)
            y1 = exposed_real_df["Y"].to_numpy()
            unexposed_real_df = real_df[real_df["T"] == 0].reset_index(drop=True)
            y0 = unexposed_real_df["Y"].to_numpy()
            real_mean = y1.mean() - y0.mean()
            real_mean_list.append(real_mean)

            ##### real data com mle estimation #####
            real_xt = real_df[["X1", "X2", "X3", "T"]].to_numpy()
            real_model = LogisticRegression(C=10000)
            real_model.fit(real_xt, y)

            real_bias_list.append(real_model.intercept_)
            real_coef_list.append(real_model.coef_)
            real_ate_list.append(
                (real_model.predict_proba(X_with_T[:,1:])[:,1] - real_model.predict_proba(X_with_T_reverse[:,1:])[:,1]).mean()
            )

            ##### real ipw estimation #####
            q1 = exposed_real_df["propensity"].to_numpy()
            q0 = 1 - unexposed_real_df["propensity"].to_numpy()
            ipw_mean = (y1 / q1).sum()/N - (y0 / q0).sum()/N
            ipw_mean_list.append(ipw_mean)

            ##### real data ipw estimation #####
            total_q = q * t + (1-q)*(1-t)
            ipw_model = LogisticRegression(C=10000)
            ipw_model.fit(real_xt, y, sample_weight=1/total_q)

            ipw_bias_list.append(ipw_model.intercept_)
            ipw_coef_list.append(ipw_model.coef_)
            ipw_ate_list.append(
                (ipw_model.predict_proba(X_with_T[:,1:])[:,1] - ipw_model.predict_proba(X_with_T_reverse[:,1:])[:,1]).mean()
            )

        random_coef_arr = np.concatenate(random_coef_list, 0)
        real_coef_arr = np.concatenate(real_coef_list, 0)
        ipw_coef_arr = np.concatenate(ipw_coef_list, 0)

        random_bias_arr = np.concatenate(random_bias_list, 0)
        real_bias_arr = np.concatenate(real_bias_list, 0)
        ipw_bias_arr = np.concatenate(ipw_bias_list, 0)

        random_ate_arr = np.array(random_ate_list)
        real_ate_arr = np.array(real_ate_list)
        ipw_ate_arr = np.array(ipw_ate_list)

        random_mean_arr = np.array(random_mean_list)
        real_mean_arr = np.array(real_mean_list)
        ipw_mean_arr = np.array(ipw_mean_list)


        true_ate = (cum_p_star - cum_p_star_reverse) / (repeat_num * N)


        print(f"x_dist: {x_dist} / N={N} / theta_loc={theta_loc} / ")
        print()
        print(f"q_bar : {np.mean(q_list).round(4)}\n")

        print(f"True ATE : \n{true_ate}")
        print(f"True Coef : \n{beta_XT_to_Y}\n")

        print(f"Random Mean ATE : \n{[(random_mean_arr - true_ate).mean().round(4)]+[random_mean_arr.var().round(4)]}")
        print(f"Real Mean ATE : \n{[(real_mean_arr - true_ate).mean().round(4)]+[real_mean_arr.var().round(4)]}")
        print(f"IPW Mean ATE : \n{[(ipw_mean_arr - true_ate).mean().round(4)]+[ipw_mean_arr.var().round(4)]}")

        print(f"Random COM ATE : \n{[(random_ate_arr - true_ate).mean().round(4)]+[random_ate_arr.var().round(4)]}")
        print(f"""COM Random Coef\n{
            [[(random_bias_arr - beta_XT_to_Y[0]).mean().round(4)]+[random_bias_arr.var().round(4)]] + 
            [[(random_coef_arr[:,i] - beta_XT_to_Y[i+1]).mean().round(4), random_coef_arr[:,i].var().round(4)] for i in range(len(random_coef_arr[0]))]
        }\n""")

        print(f"Real COM ATE : \n{[(real_ate_arr - true_ate).mean().round(4)]+[real_ate_arr.var().round(4)]}")
        print(f"""COM Real Coef\n{
            [[(real_bias_arr - beta_XT_to_Y[0]).mean().round(4)]+[real_bias_arr.var().round(4)]] + 
            [[(real_coef_arr[:,i] - beta_XT_to_Y[i+1]).mean().round(4), real_coef_arr[:,i].var().round(4)] for i in range(len(real_coef_arr[0]))]
        }\n""")

        print(f"IPW COM ATE : \n{[(ipw_ate_arr - true_ate).mean().round(4)]+[ipw_ate_arr.var().round(4)]}")
        print(f"""COM IPW Coef\n{
            [[(ipw_bias_arr - beta_XT_to_Y[0]).mean().round(4)]+[ipw_bias_arr.var().round(4)]] + 
            [[(ipw_coef_arr[:,i] - beta_XT_to_Y[i+1]).mean().round(4), ipw_coef_arr[:,i].var().round(4)] for i in range(len(ipw_coef_arr[0]))]
        }\n""")

# %%
