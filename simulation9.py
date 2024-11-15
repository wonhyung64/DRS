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
t_coef = np.array([3, 2])
y1_coef = np.array([-1, -2])
y0_coef = np.array([-1, 2])
covariate_num = len(y1_coef)
if covariate_num - len(y0_coef):
    raise ValueError("covariate num is different")

all_x = np.array(list(itertools.product(x_values, repeat=covariate_num)))

mle_all_auc_list, ipw_all_auc_list = [], []
mle_x1_auc_list, ipw_x1_auc_list = [], []

mle_all_coef1_list, ipw_all_coef1_list = [], []
mle_all_coef2_list, ipw_all_coef2_list = [], []

mle_x1_coef1_list, ipw_x1_coef1_list = [], []

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
    x_obs_x1 = sampled_x[T==1][:, 0:1]
    ps_obs = prob_t[T==1]

    """MLE all ESTIMATION"""
    mle_all_model = LogisticRegression(penalty=None, fit_intercept=False)
    mle_all_model.fit(x_obs_all, y_obs)
    mle_all_coef = mle_all_model.coef_
    mle_all_pred = mle_all_model.predict_proba(x_obs_all)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_obs, mle_all_pred, pos_label=1)
    mle_all_auc = auc(fpr, tpr)
    mle_all_auc_list.append(mle_all_auc)
    mle_all_coef1_list.append(mle_all_coef[0,0])
    mle_all_coef2_list.append(mle_all_coef[0,1])

    """IPW all ESTIMATION"""
    ipw_all_model = LogisticRegression(penalty=None, fit_intercept=False)
    ipw_all_model.fit(x_obs_all, y_obs, sample_weight=1/ps_obs)
    ipw_all_coef = ipw_all_model.coef_
    ipw_all_pred = ipw_all_model.predict_proba(x_obs_all)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_obs, ipw_all_pred, pos_label=1)
    ipw_all_auc = auc(fpr, tpr)
    ipw_all_auc_list.append(ipw_all_auc)
    ipw_all_coef1_list.append(ipw_all_coef[0,0])
    ipw_all_coef2_list.append(ipw_all_coef[0,1])

    """MLE x1 ESTIMATION"""
    mle_x1_model = LogisticRegression(penalty=None, fit_intercept=False)
    mle_x1_model.fit(x_obs_x1, y_obs)
    mle_x1_coef = mle_x1_model.coef_
    mle_x1_pred = mle_x1_model.predict_proba(x_obs_x1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_obs, mle_x1_pred, pos_label=1)
    mle_x1_auc = auc(fpr, tpr)
    mle_x1_auc_list.append(mle_x1_auc)
    mle_x1_coef1_list.append(mle_x1_coef[0,0])

    """IPW x1 ESTIMATION"""
    ipw_x1_model = LogisticRegression(penalty=None, fit_intercept=False)
    ipw_x1_model.fit(x_obs_x1, y_obs, sample_weight=1/ps_obs)
    ipw_x1_coef = ipw_x1_model.coef_
    ipw_x1_pred = ipw_x1_model.predict_proba(x_obs_x1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_obs, ipw_x1_pred, pos_label=1)
    ipw_x1_auc = auc(fpr, tpr)
    ipw_x1_auc_list.append(ipw_x1_auc)
    ipw_x1_coef1_list.append(ipw_x1_coef[0,0])

#%%
print(
    (np.array(mle_all_coef1_list) - y1_coef[0]).mean().round(4),
    np.var(mle_all_coef1_list).round(4),
    (np.array(mle_all_coef2_list) - y1_coef[1]).mean().round(4),
    np.var(mle_all_coef2_list).round(4),
    np.mean(mle_all_auc_list).round(4),
    np.std(mle_all_auc_list).round(4),
)

print(
    (np.array(ipw_all_coef1_list) - y1_coef[0]).mean().round(4),
    np.var(ipw_all_coef1_list).round(4),
    (np.array(ipw_all_coef2_list) - y1_coef[1]).mean().round(4),
    np.var(ipw_all_coef2_list).round(4),
    np.mean(ipw_all_auc_list).round(4),
    np.std(ipw_all_auc_list).round(4),
)

print(
    (np.array(mle_x1_coef1_list) - y1_coef[0]).mean().round(4),
    np.var(mle_x1_coef1_list).round(4),
    "-",
    "-",
    np.mean(mle_x1_auc_list).round(4),
    np.std(mle_x1_auc_list).round(4),
)

print(
    (np.array(ipw_x1_coef1_list) - y1_coef[0]).mean().round(4),
    np.var(ipw_x1_coef1_list).round(4),
    "-",
    "-",
    np.mean(ipw_x1_auc_list).round(4),
    np.std(ipw_x1_auc_list).round(4),
)
# %%
