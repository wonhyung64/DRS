#%%
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


#%%
n = 1000
x_values = [1, -1]
t_coef = np.array([3,2])
y1_coef = np.array([-1,-2])
y0_coef = np.array([-1,2])
covariate_num = len(y1_coef)
if covariate_num - len(y0_coef):
    raise ValueError("covariate num is different")

all_x = np.array(list(itertools.product(x_values, repeat=covariate_num)))


# x_i가 [1, -1] 중 하나의 원소일 때 모든 가능한 샘플 생성
def logistic(x, coef, bias=0.):
    return 1 / (1 + np.exp(np.sum(-(x*coef+bias), -1)))

EY1 = 0.
EY0 = 0.
for x in all_x:
    EY1 += logistic(x, y1_coef)/(2**covariate_num)
    EY0 += logistic(x, y0_coef)/(2**covariate_num)
        
true_ate = (EY1 - EY0) ### TRUE ATE


repeat_num = 10000

com_ATE_list = []
gcom_ATE_list = []
mean_ATE_list = []
ipw_ATE_list = []


for repeat_seed in tqdm(range(1, repeat_num+1)):
    np.random.seed(repeat_seed)

    """DATA GENERATION"""
    sampled_x = np.random.choice([-1, 1], [n,covariate_num])
    prob_t = logistic(sampled_x, t_coef)
    T = np.random.binomial(1, prob_t)
    prob_y1 = logistic(sampled_x, y1_coef)
    prob_y0 = logistic(sampled_x, y0_coef)
    Y1 = np.random.binomial(1, prob_y1)
    Y0 = np.random.binomial(1, prob_y0)
    Y = Y1 * T + Y0 * (1-T)


    """COM ESTIMATION"""
    model = LogisticRegression(penalty=None, fit_intercept=False)
    sampled_xt = np.concatenate([sampled_x, T.reshape(-1, 1)], -1)
    model.fit(sampled_xt, Y)
    hat_y_coef = model.coef_
    
    hat_EY1, hat_EY0 = 0., 0.
    for x in all_x:
        hat_EY1 += logistic(x, hat_y_coef[:,:-1], hat_y_coef[:, -1])/(2**covariate_num)
        hat_EY0 += logistic(x, hat_y_coef[:,:-1])/(2**covariate_num)

    com_ATE = (hat_EY1 - hat_EY0).item()


    """GCOM ESTIMATION"""
    y1_obs = Y[T == 1]
    x1_obs = sampled_x[T == 1]
    model1 = LogisticRegression(penalty=None, fit_intercept=False)
    model1.fit(x1_obs, y1_obs)

    y0_obs = Y[T == 0]
    x0_obs = sampled_x[T == 0]
    model0 = LogisticRegression(penalty=None, fit_intercept=False)
    model0.fit(x0_obs, y0_obs)
    
    hat_y1_coef = model1.coef_
    hat_y0_coef = model0.coef_
    

    hat_EY1, hat_EY0 = 0., 0.
    for x in all_x:
        hat_EY1 += logistic(x, hat_y1_coef)/(2**covariate_num)
        hat_EY0 += logistic(x, hat_y0_coef)/(2**covariate_num)

    gcom_ATE = (hat_EY1 - hat_EY0).item()


    """MEAN CATE"""
    mean_ATE = y1_obs.mean() - y0_obs.mean()

    """IPW CATE"""
    y1_ps = prob_t[T==1]
    y0_ps = prob_t[T==0]
    ipw_ATE = (y1_obs/y1_ps).sum()/n - (y0_obs/(1-y0_ps)).sum()/n

    
    com_ATE_list.append(com_ATE)
    gcom_ATE_list.append(gcom_ATE)
    mean_ATE_list.append(mean_ATE)
    ipw_ATE_list.append(ipw_ATE)

print(true_ate)
print((np.array(com_ATE_list) - true_ate).mean().round(4), np.var(com_ATE_list).round(4))
print((np.array(gcom_ATE_list) - true_ate).mean().round(4), np.var(gcom_ATE_list).round(4))
print((np.array(mean_ATE_list) - true_ate).mean().round(4), np.var(mean_ATE_list).round(4))
print((np.array(ipw_ATE_list) - true_ate).mean().round(4), np.var(ipw_ATE_list).round(4))


# %%
