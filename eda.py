#%%
import os 
import json
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


# %%
data_dir = "./data"

os.listdir(f"{data_dir}/yahoo_r3")

train_dir = f"{data_dir}/yahoo_r3/ydata-ymusic-rating-study-v1_0-train.txt"

with open(train_dir, "r") as f:
    train_raw = f.readlines()


#%% popularity bias in yahoo r3
dataset_path = "/root/won/DRS/data/yahoo_r3/implicit"
train_data_path: str = dataset_path + '/train.csv'
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]

item_dist = train_df["item_id"].value_counts().reset_index(drop=True)

plt.bar(x=item_dist.index.values, height=item_dist.values)
plt.xlabel("Item", fontsize=16)
plt.ylabel("# of Interactions", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#%% popularity bias in coat
dataset_path = "/root/won/DRS/data/coat"
train_data_path: str = dataset_path + '/train.csv'
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]

item_dist = train_df["item_id"].value_counts().reset_index(drop=True)

plt.bar(x=item_dist.index.values, height=item_dist.values)
plt.xlabel("Item", fontsize=16)
plt.ylabel("# of Interactions", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#%% exposure bias in yahoo r3 and coat
dataset_path = "/root/won/DRS/data/yahoo_r3/implicit"
train_data_path: str = dataset_path + '/train.csv'
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]

pos_yahoo = train_df["user_id"].value_counts().reset_index(drop=True)
neg_yahoo = 1000 - pos_yahoo

dataset_path = "/root/won/DRS/data/coat"
train_data_path: str = dataset_path + '/train.csv'
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]

pos_coat = train_df["user_id"].value_counts().reset_index(drop=True)
neg_coat = 300 - pos_coat

plt.boxplot(pos_yahoo)
plt.boxplot(pos_coat)


# %%
X1=[1,3]
data1 = [np.mean(pos_yahoo), np.mean(pos_coat)]
X2=[1+0.5,3+0.5]
data2 = [np.mean(neg_yahoo), np.mean(neg_coat)]
ticklabel=['Yahoo R3','Coat']


# setting figure
plt.rcParams['lines.linewidth'] = 4.0
plt.rcParams['boxplot.flierprops.markersize'] = 10

# subplot
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(8,6))

# making yaxis grid
ax1.yaxis.grid()
ax2.yaxis.grid()

#making barplot for each subplot ax1 and ax2
ax1.bar(X1, data1, width=0.5, label="observed")
ax1.bar(X2, data2, width=0.5, label="unobserved")

ax2.bar(X1, data1, width=0.5)
ax2.bar(X2, data2, width=0.5)

plt.xticks(size=16)
plt.yticks(size=16)

ax1.set_ylim(200, 1100)
ax2.set_ylim(0, 30)

ax1.set_yticks([200, 500, 800, 1100])
ax2.set_yticks([0, 10, 20, 30])

ax1.set_xticks([1+0.25, 3+0.25])
ax2.set_xticks([1+0.25, 3+0.25])

ax1.set_ylabel("")
ax2.set_ylabel("")

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.text(0., 0.50, "# of Interactions", va='center', rotation = 'vertical', fontsize = 16)
ax1.get_xaxis().set_visible(False)

labels = ax1.set_yticklabels(['200', '500', '800', '1100'], fontsize = 16)
labels = ax2.set_yticklabels(['0', '10','20', '30'], fontsize = 16)
# how big to make the diagonal lines in axes coordinates
d = .7    
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

labels = ax2.set_xticks([1+0.25, 3+0.25], ticklabel, fontsize=16, rotation=0)

ax1.set_xlabel("")
ax1.legend(fontsize=16)

# %%
