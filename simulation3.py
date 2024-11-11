#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


#%%
n = 100000
t_coef = 3
y1_coef = -2
y0_coef = 1

EY1 = (1 / (1 + np.exp(-y1_coef*1)) +  1 / (1 + np.exp(-y1_coef*(-1))))/2
EY0= (1 / (1 + np.exp(-y0_coef*1)) +  1 / (1 + np.exp(-y0_coef*(-1))))/2
true_ate = (EY1 - EY0) ### TRUE ATE
repeat_num = 1000

com_mean_list = []
real_mean_list = []
ipw_mean_list = []


for repeat_seed in tqdm(range(1, repeat_num+1)):
    np.random.seed(repeat_seed)

    x = np.random.choice([-1, 1], n)
    prob_t = 1 / (1 + np.exp(-t_coef*x))
    T = np.random.binomial(1, prob_t)
    prob_y1 = 1 / (1 + np.exp(y1_coef*x))
    prob_y0 = 1 / (1 + np.exp(-y0_coef*x))
    Y1 = np.random.binomial(1, prob_y1)
    Y0 = np.random.binomial(1, prob_y0)
    Y = Y1 * T + Y0 * (1-T)

    # xt = np.array([x*T, x]).T
    xt = np.array([x, T]).T
    model = LogisticRegression(penalty=None, fit_intercept=False)
    model.fit(xt, Y)

    b1 = model.coef_[0,0]
    gamma = model.coef_[0,1]
    # x = 1, T = 1
    e11 = 1/(1+ np.exp(-(b1*1 + gamma*1)))
    # x = -1, T = 1
    e12 = 1/(1+ np.exp(-(b1*(-1) + gamma*1)))
    # x = 1, T = 0
    e21 = 1/(1+ np.exp(-(b1*(1))))
    # x = -1, T = 0
    e22 = 1/(1+ np.exp(-b1*(-1)))

    com_mean = (e11 + e12)/2 - (e21 + e22)/2

    all_table = np.stack([x, Y1, Y0, T, Y, prob_t], -1)
    df = pd.DataFrame(all_table)
    df.columns = ["x", "y1", "y0", "t", "y", "ps"]

    real_mean = df[df["t"] == 1]["y"].mean() - df[df["t"] == 0]["y"].mean()
    ipw_mean = (df[df["t"] == 1]["y"]/df[df["t"] == 1]["ps"]).mean() - (df[df["t"] == 0]["y"]/(1-df[df["t"] == 0]["ps"])).mean()

    com_mean_list.append(com_mean)
    real_mean_list.append(real_mean)
    ipw_mean_list.append(ipw_mean)


print(true_ate)
print((np.array(com_mean_list) - true_ate).mean(), np.var(com_mean_list))
print((np.array(real_mean_list) - true_ate).mean(), np.var(real_mean_list))
print((np.array(ipw_mean_list) - true_ate).mean(), np.var(ipw_mean_list))

#%%
df[(df["t"] == 1) & (df["x"] == 1)]["y"].mean()
df[(df["t"] == 1) & (df["x"] == -1)]["y"].mean()
df[(df["t"] == 0) & (df["x"] == 1)]["y"].mean()
df[(df["t"] == 0) & (df["x"] == -1)]["y"].mean()

df[(df["t"] == 1) & (df["x"] == 1)].count()
df[(df["t"] == 1) & (df["x"] == -1)].count()
df[(df["t"] == 0) & (df["x"] == 1)].count()
df[(df["t"] == 0) & (df["x"] == -1)].count()

# %%
