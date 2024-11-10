#%%
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


#%% 설정
repeat_num = 1000

d = 3 # feature dimension
treatment_effect = 2.0
theta_X_to_T_ = np.array([2.5, 1.0, -1.5, 0.5])   # X -> T로 가는 경로의 계수
beta_XT_to_Y = np.array([2.2, 1.5, -2.0, 0.5, treatment_effect])  # X와 T -> Y로 가는 경로의 계수 (마지막은 T의 계수

theta_loc_list1 = [0., 30.,] #normal 10^4
theta_loc_list2 = [0., 0.32, 1.5] #uniform 10^4
theta_loc_list3 = [0., 0.3, 1.5, 3.] #uniform 10^6

for (x_dist, N ,theta_loc_list) in [
    ("normal", 10000, theta_loc_list1),
    ("uniform", 10000, theta_loc_list2),
    ("uniform", 1000000, theta_loc_list3),
    ]:

    for theta_loc in theta_loc_list:
        print(f"theta_loc : {theta_loc}")

        theta_X_to_T = theta_X_to_T_ - theta_loc
        beta_XT_to_Y = np.array(list(-theta_X_to_T) + [treatment_effect])

        random_bias_list = []
        real_bias_list = []
        ipw_bias_list = []
        random_coef_list = []
        real_coef_list = []
        ipw_coef_list = []
        q_list = []

        for repeat_seed in tqdm(range(1, repeat_num+1)):
            np.random.seed(repeat_seed)

            # 1. 독립 변수 X 생성
            if x_dist == "normal":
                x = np.random.normal(0, 1, (N, d))  # Normal(0,1)
            elif x_dist == "uniform":
                x = np.random.uniform(0, 1, (N, d))  # [0,1]  Uniform
            design_x = np.concatenate([np.ones([N,1]), x], axis=1)

            # 종속 변수 Y 생성 (X와 T의 영향을 받음)
            # Y는 X와 T의 선형 결합에 대한 로지스틱 확률로 결정되는 이진 변수
            dummy = np.ones(N).reshape(-1, 1)

            X_with_T = np.hstack((design_x, dummy))  
            logit_p_star = X_with_T.dot(beta_XT_to_Y)
            p_star = 1 / (1 + np.exp(-logit_p_star))
            y_forward = np.random.binomial(1, p_star)

            X_with_T_reverse = np.hstack((design_x, 1 - dummy))
            logit_p_star_reverse = X_with_T_reverse.dot(beta_XT_to_Y)
            p_star_reverse = 1 / (1 + np.exp(-logit_p_star_reverse))
            y_reverse = np.random.binomial(1, p_star_reverse)


            ##### randomized data generation ######
            q_star = np.ones(N) * 1/2
            t_star = np.random.binomial(1, q_star)
            y_star = y_forward * t_star + y_reverse * (1-t_star)

            random_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
            random_df['T'] = t_star
            random_df['Y'] = y_star

            exposed_random_df = random_df[random_df["T"] == 1].reset_index(drop=True)
            x_treated = exposed_random_df[["X1", "X2", "X3"]].to_numpy()
            y_true = exposed_random_df["Y"].to_numpy()


            ##### randomized data fitting #####
            random_model = LogisticRegression(C=10000)
            random_model.fit(x_treated, y_true)
            random_bias_list.append(random_model.intercept_)
            random_coef_list.append(random_model.coef_)


            ##### partial data generation #####
            logit_q = design_x.dot(theta_X_to_T)
            q = 1 / (1 + np.exp(-logit_q))
            t = np.random.binomial(1, q)
            y = y_forward * t + y_reverse * (1-t)

            q_list.append(q.mean())

            real_df = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(d)])
            real_df['T'] = t
            real_df['Y'] = y
            real_df['propensity'] = q

            exposed_real_df = real_df[real_df["T"] == 1].reset_index(drop=True)
            x_treated = exposed_real_df[["X1", "X2", "X3"]].to_numpy()
            y_true = exposed_real_df["Y"].to_numpy()
            exposed_q = exposed_real_df["propensity"].to_numpy()


            ##### partial data fitting #####
            real_model = LogisticRegression(C=10000)
            real_model.fit(x_treated, y_true)
            real_bias_list.append(real_model.intercept_)
            real_coef_list.append(real_model.coef_)


            ##### partial data ipw fitting #####
            ipw_model = LogisticRegression(C=10000)
            ipw_model.fit(x_treated, y_true, sample_weight=1/(exposed_q))
            ipw_bias_list.append(ipw_model.intercept_)
            ipw_coef_list.append(ipw_model.coef_)


        random_coef_arr = np.concatenate(random_coef_list, 0)
        real_coef_arr = np.concatenate(real_coef_list, 0)
        ipw_coef_arr = np.concatenate(ipw_coef_list, 0)

        random_bias_arr = np.concatenate(random_bias_list, 0)
        real_bias_arr = np.concatenate(real_bias_list, 0)
        ipw_bias_arr = np.concatenate(ipw_bias_list, 0)

        true_beta = [beta_XT_to_Y[0] + beta_XT_to_Y[-1]] + list(beta_XT_to_Y[1:-1])


        print(f"x_dist: {x_dist} / N={N} / theta_loc={theta_loc} / ")
        print()
        print(f"q_bar : {np.mean(q_list).round(4)}\n")
        print(f"True Coef : \n{true_beta}\n")

        random_coef_ = [[(random_bias_arr - true_beta[0]).mean().round(4)]+[random_bias_arr.var().round(4)]] + \
            [[(random_coef_arr[:,i] - true_beta[i+1]).mean().round(4), random_coef_arr[:,i].var().round(4)] for i in range(d)]
        random_coef = list(itertools.chain(*random_coef_))
        print("Random Coef : ")
        print('$\\hat{\\mathcal{L}}_{rand}$' + "".join([f' & ${format(value, ".4f")}$' for value in random_coef]) + " \\\\")
        print()

        real_coef_ = [[(real_bias_arr - true_beta[0]).mean().round(4)]+[real_bias_arr.var().round(4)]] + \
            [[(real_coef_arr[:,i] - true_beta[i+1]).mean().round(4), real_coef_arr[:,i].var().round(4)] for i in range(d)]
        real_coef = list(itertools.chain(*real_coef_))
        print("Real Coef : ")
        print('$\\hat{\\mathcal{L}}_{real}$' + "".join([f' & ${format(value, ".4f")}$' for value in real_coef]) + " \\\\")
        print()
        
        ipw_coef_ = [[(ipw_bias_arr - true_beta[0]).mean().round(4)]+[ipw_bias_arr.var().round(4)]] + \
            [[(ipw_coef_arr[:,i] - true_beta[i+1]).mean().round(4), ipw_coef_arr[:,i].var().round(4)] for i in range(d)]
        ipw_coef = list(itertools.chain(*ipw_coef_))
        print("IPW Coef : ")
        print('$\\hat{\\mathcal{L}}_{ipw}$' + "".join([f' & ${format(value, ".4f")}$' for value in ipw_coef]) + " \\\\")
        print()

# %%