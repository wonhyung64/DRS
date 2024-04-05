#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import torch.nn.functional as F
from datetime import datetime

from module.model import NCF
from module.metric import ndcg_func
from module.utils import binarize, shuffle

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def contrastive_loss(user_embed, aug_user_embed, scale=1.):
    batch_size = user_embed.shape[0]
    org_norm = F.normalize(user_embed, p=2, dim=1)
    aug_norm = F.normalize(aug_user_embed, p=2, dim=1)
    pred = F.linear(org_norm, aug_norm) / scale
    pos_label = torch.eye(batch_size).to(user_embed.device)
    neg_label = 1 - pos_label
    pos_feat = (pred.exp() * pos_label).sum(dim=-1)
    neg_feat = (pred.exp() * neg_label).sum(dim=-1)

    return -torch.log(pos_feat / (pos_feat + neg_feat)).mean()


def triplet_loss(anchor_user_embed, pos_user_embed, neg_user_embed, dist='sqeuclidean', margin='maxplus'):
    pos_dist = torch.square(anchor_user_embed - pos_user_embed)
    neg_dist = torch.square(anchor_user_embed - neg_user_embed)

    if dist == 'euclidean':
        pos_dist = torch.sqrt(torch.sum(pos_dist, dim=-1))
        neg_dist = torch.sqrt(torch.sum(neg_dist, dim=-1))
    elif dist == 'sqeuclidean':
        pos_dist = torch.sum(pos_dist, axis=-1)
        neg_dist = torch.sum(neg_dist, axis=-1)

    loss = pos_dist - neg_dist

    if margin == 'maxplus':
        loss = torch.maximum(torch.tensor(0.0), 1 + loss)
    elif margin == 'softplus':
        loss = torch.log(1 + torch.exp(loss))

    return torch.mean(loss)


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--balance-param", type=float, default=1.5)
parser.add_argument("--temperature", type=float, default=1.)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset-name", type=str, default="yahoo_r3")
parser.add_argument("--contrast-pair", type=str, default="both")


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
temperature = args.temperature
data_dir = args.data_dir
dataset_name = args.dataset_name
contrast_pair = args.contrast_pair


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
data_set_dir = os.path.join(data_dir, dataset_name)
train_file = os.path.join(data_set_dir, "ydata-ymusic-rating-study-v1_0-train.txt")
test_file = os.path.join(data_set_dir, "ydata-ymusic-rating-study-v1_0-test.txt")

x_train = []
with open(train_file, "r") as f:
    for line in f:
        x_train.append(line.strip().split())
# <user_id> <song id> <rating>
x_train = np.array(x_train).astype(int)

x_test = []
with open(test_file, "r") as f:
    for line in f:
        x_test.append(line.strip().split())
# <user_id> <song id> <rating>
x_test = np.array(x_test).astype(int)

print("===>Load from {} data set<===".format(dataset_name))
print("[train] num data:", x_train.shape[0])
print("[test]  num data:", x_test.shape[0])

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

train_user_indices = x_train.copy()[:, 0] - 1
pos_user_samples_ = np.load("./assets/pos_user_samples.npy")
neg_user_samples_ = np.load("./assets/neg_user_samples.npy")
pos_user_samples = pos_user_samples_[train_user_indices]
neg_user_samples = neg_user_samples_[train_user_indices]
user_pos_neg = np.stack([pos_user_samples, neg_user_samples], axis=-1)

train_item_indices = x_train.copy()[:, 1] - 1
pos_item_samples_ = np.load("./assets/pos_item_samples.npy")
neg_item_samples_ = np.load("./assets/neg_item_samples.npy")
pos_item_samples = pos_item_samples_[train_item_indices]
neg_item_samples = neg_item_samples_[train_item_indices]
item_pos_neg = np.stack([pos_item_samples, neg_item_samples], axis=-1)

total_samples = np.concatenate([x_train, user_pos_neg, item_pos_neg], axis=-1)
total_samples, y_train = shuffle(total_samples, y_train)
x_train, user_pos_neg, item_pos_neg = total_samples[:,0:2], total_samples[:,2:4], total_samples[:,4:6]

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print("# user: {}, # item: {}".format(num_users, num_items))

y_train = binarize(y_train)
y_test = binarize(y_test)

num_sample = len(x_train)
batch_size = 64
total_batch = num_sample // batch_size

#%% TRAIN INITIAILIZE
all_idx = np.arange(num_sample)
np.random.shuffle(all_idx)

user_duplicate_ratios = []
item_duplicate_ratios = []
from tqdm import tqdm
for idx in tqdm(range(total_batch)):
    # mini-batch training
    selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
    sub_x = x_train[selected_idx]
    org_x = torch.LongTensor(sub_x - 1).to(device)

    sub_y = y_train[selected_idx]
    sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

    pos_x = np.stack([user_pos_neg[selected_idx,0], item_pos_neg[selected_idx,0]], axis=-1)
    pos_x = torch.LongTensor(pos_x - 1).to(device)
    
    user_pos = pos_x[:,0]
    for u in user_pos:
        user_duplicate_ratio = ((user_pos == u).sum() - 1).item() / batch_size
        user_duplicate_ratios.append(user_duplicate_ratio)

    item_pos = pos_x[:,1]
    for i in item_pos:
        item_duplicate_ratio = ((item_pos == i).sum() - 1).item() / batch_size
        item_duplicate_ratios.append(item_duplicate_ratio)

import matplotlib.pyplot as plt
plt.hist(user_duplicate_ratios, bins=20)
np.mean(user_duplicate_ratios)
np.std(user_duplicate_ratios)
plt.hist(item_duplicate_ratios, bins=15)
np.mean(item_duplicate_ratios)
np.std(item_duplicate_ratios)


#%%
pos_user_nums = []
for u in tqdm(range(1, num_users+1)):
    pos_user_nums.append(sum(user_pos_neg[:, 0] == u))

pos_item_nums = []
for i in tqdm(range(1, num_items+1)):
    pos_item_nums.append(sum(item_pos_neg[:, 0] == i))
import seaborn as sns
sns.barplot(pos_item_nums)
plt.bar(range(1,15401), pos_user_nums)
plt.bar(range(1,1001), pos_item_nums)

plt.boxplot(pos_item_nums)
plt.boxplot(pos_user_nums)
