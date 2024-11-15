#%%
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression


def logistic(x, coef, bias=0.):
    return 1 / (1 + np.exp(np.sum(-(x*coef+bias), -1)))


#%%
n = 100000
repeat_num = 1000
x_values = [1, -1]
covariate_num = 15

np.random.seed(0)
t_coef = np.random.choice(np.arange(-5, 5), covariate_num)
y1_coef = np.random.choice(np.arange(-5, 5), covariate_num)
y0_coef = np.random.choice(np.arange(-5, 5), covariate_num)

mle_auc_list, ipw_auc_list = [], []
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

    y_obs = Y[T==1]
    x_obs_all = sampled_x[T==1]
    ps_obs = prob_t[T==1]

    mle_auc_per, ipw_auc_per = [], []
    for cov_idx in [3, 6, 9, 12, 15]:

        """MLE all ESTIMATION"""
        mle_model = LogisticRegression(penalty=None, fit_intercept=False)
        mle_model.fit(x_obs_all[:, :cov_idx], y_obs)
        mle_pred = mle_model.predict_proba(x_obs_all[:, :cov_idx])[:, 1]
        fpr, tpr, thresholds = roc_curve(y_obs, mle_pred, pos_label=1)
        mle_auc = auc(fpr, tpr)

        """IPW all ESTIMATION"""
        ipw_model = LogisticRegression(penalty=None, fit_intercept=False)
        ipw_model.fit(x_obs_all[:, :cov_idx], y_obs, sample_weight=1/ps_obs)
        ipw_pred = ipw_model.predict_proba(x_obs_all[:, :cov_idx])[:, 1]
        fpr, tpr, thresholds = roc_curve(y_obs, ipw_pred, pos_label=1)
        ipw_auc = auc(fpr, tpr)

        mle_auc_per.append(mle_auc)
        ipw_auc_per.append(ipw_auc)

    mle_auc_list.append(mle_auc_per)
    ipw_auc_list.append(ipw_auc_per)

mle_auc_arr = np.array(mle_auc_list)
ipw_auc_arr = np.array(ipw_auc_list)


result_str = ""
for i in range(mle_auc_arr.shape[1]):
    result_str += f"& ${mle_auc_arr[:,i].mean().round(4)}_{{\pm {mle_auc_arr[:,i].std().round(4)}}}$ "
print(result_str + r" \\")

result_str = ""
for i in range(ipw_auc_arr.shape[1]):
    result_str += f"& ${ipw_auc_arr[:,i].mean().round(4)}_{{\pm {ipw_auc_arr[:,i].std().round(4)}}}$ "
print(result_str + r" \\")

#%%
gamma = 3-0.0000001
np.exp(-1+gamma) / (1 + np.exp(-1+gamma))**2  - np.exp(1+gamma) / (1 + np.exp(1+gamma))**2
gamma = 0
np.exp(-1+gamma) / (1 + np.exp(-1+gamma))**2  - np.exp(1+gamma) / (1 + np.exp(1+gamma))**2