#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datetime import datetime

from module.model import NCF, MF
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import binarize
from module.loss import contrastive_loss, hard_contrastive_loss
from module.similarity import compute_sim_matrix, seperate_pos_neg_interactions, corr_sim, cosine_sim
from module.dataset import load_dataset

#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--balance-param", type=float, default=1.)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset-name", type=str, default="coat")
parser.add_argument("--contrast-pair", type=str, default="both")
parser.add_argument("--base-model", type=str, default="ncf")
parser.add_argument("--sim-measure", type=str, default="cosine")
parser.add_argument("--temperature", type=float, default=2.)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


embedding_k = args.embedding_k
lr = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
balance_param = args.balance_param
data_dir = args.data_dir
dataset_name = args.dataset_name
contrast_pair = args.contrast_pair
base_model = args.base_model
sim_measure = args.sim_measure
temperature = args.temperature


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
save_dir = f"./weights/expt_{expt_num}"
os.makedirs(f"{save_dir}", exist_ok=True)

np.random.seed(random_seed)
torch.manual_seed(random_seed)


#%% DATA LOADER
dataset_dir = os.path.join(data_dir, dataset_name)
x_train, x_test = load_dataset(dataset_name, dataset_dir)

print("===>Load from {} data set<===".format(dataset_name))
print("[train] num data:", x_train.shape[0])
print("[test]  num data:", x_test.shape[0])

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

if dataset_name != "kuairec":
    y_train = binarize(y_train)
    y_test = binarize(y_test)

num_sample = len(x_train)
total_batch = num_sample // batch_size

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print("# user: {}, # item: {}".format(num_users, num_items))



#%% User/Item similarity
if sim_measure == "corr":
    user_user_sim, item_item_sim = corr_sim(x_train, y_train, num_users, num_items)
elif sim_measure == "cosine":
    user_user_sim, item_item_sim = cosine_sim(x_train, y_train, num_users, num_items)


user_idx = (user_user_sim != 0).sum(-1).argmax()
alpha = user_user_sim[user_idx,:]
topk_indices = alpha.topk(10).indices.numpy()
alpha[topk_indices] = 1.
alpha[126] = 1.
c = ["green" if i in topk_indices else "white" for i in range(290)]
c[126] = "red"
c[191]
alpha[191]
topk_indices
#%%
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dosnes import dosnes
import torch
from baselines.ncf.model import NCF

ncf = NCF(290, 300, 64)
ncf.load_state_dict(torch.load('/Users/wonhyung64/Github/DRS/baselines/ncf/ncf.pth'))
# ncf.load_state_dict(torch.load('/Users/wonhyung64/Github/DRS/cosine_ncf.pth'), strict=False)

items = torch.arange(0, 300)
users = torch.arange(0, 290)

_, user_embed, __ = ncf(torch.stack([users, torch.zeros_like(users)], -1))
X_embed = user_embed.detach().numpy()

# _, __, item_embed = ncf(torch.stack([torch.zeros_like(items), items], -1))
# X_embed = item_embed.detach().numpy()

# X, y = datasets.load_digits(return_X_y = True)
metric = "sqeuclidean"
model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42, max_iter = 1000)
X_embedded = model.fit_transform(X_embed)


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=c, cmap=plt.cm.Set1)
plt.title("Digits Dataset Embedded on a Sphere with metric {}".format(metric))
plt.show()


# %%
