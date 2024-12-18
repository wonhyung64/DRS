#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
from datetime import datetime

from model import MF
from metric import ndcg_func, recall_func, ap_func
from utils import binarize

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


class MF_IPS_V2(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)       
        self.propensity_model = NCF(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)


class MF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)

        return out, user_embed, item_embed


class NCF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k*2, 1, bias=True)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)

        out = self.linear_1(z_embed)

        return out, user_embed, item_embed


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-2)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../../data")
parser.add_argument("--dataset-name", type=str, default="coat")

parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--eta", type=float, default=1.)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--G", type=int, default=1)

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
data_dir = args.data_dir
dataset_name = args.dataset_name
alpha = args.alpha
beta = args.beta
eta = args.eta
gamma = args.gamma
G = args.G


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


#%% WANDB
wandb_var = wandb.init(
    project="recommender",
    config={
        "device" : device,
        "embedding_k" : embedding_k,
        "batch_size" : batch_size,
        "num_epochs" : num_epochs,
        "evaluate_interval" : evaluate_interval,
        "lr" : lr,
        "weight_decay": weight_decay,
        "top_k_list" : top_k_list,
        "random_seed" : random_seed,
        "dataset_name" : dataset_name,
        "alpha": alpha,
        "beta": beta,
        "eta": eta,
        "gamma": gamma,
        "G": G,
    }
)
wandb.run.name = f"IPSv2_{expt_num}"


#%% DATA LOADER
data_set_dir = os.path.join(data_dir, dataset_name)

if dataset_name == "yahoo_r3":
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

elif dataset_name == "coat":
    train_file = os.path.join(data_set_dir, "train.csv")
    test_file = os.path.join(data_set_dir, "test.csv")

    x_train = pd.read_csv(train_file).to_numpy()
    x_train = np.stack([x_train[:,0]+1, x_train[:,1]+1, x_train[:,2]], axis=-1)

    x_test = pd.read_csv(test_file).to_numpy()
    x_test = np.stack([x_test[:,0]+1, x_test[:,1]+1, x_test[:,2]], axis=-1)

elif dataset_name == "kuairec":
    train_file = os.path.join(data_set_dir, "data/big_matrix.csv")
    test_file = os.path.join(data_set_dir, "data/small_matrix.csv")

    x_train = pd.read_csv(train_file)
    x_train["interaction"] = x_train["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)
    x_train = x_train[["user_id", "video_id", "interaction"]].to_numpy()
    x_train = np.stack([x_train[:,0]+1, x_train[:,1]+1, x_train[:,2]], axis=-1)

    x_test = pd.read_csv(test_file)
    x_test["interaction"] = x_test["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)
    x_test = x_test[["user_id", "video_id", "interaction"]].to_numpy()
    x_test = np.stack([x_test[:,0]+1, x_test[:,1]+1, x_test[:,2]], axis=-1)
    
elif dataset_name == "ml-1m":
    x_train = np.load(f"{data_set_dir}/train.npy")
    x_test = np.load(f"{data_set_dir}/test.npy")

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


#%% TRAIN
model = MF_IPS_V2(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()
num_users * num_items
obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
# generate all counterfactuals and factuals
x_all = generate_total_sample(num_users, num_items)

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    ul_idxs = np.arange(x_all.shape[0]) # all
    np.random.shuffle(ul_idxs)

    epoch_total_loss = 0.
    epoch_rec_loss = 0.

    for idx in range(total_batch):

        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx] - 1
        sub_x = torch.LongTensor(sub_x).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
        x_sampled = x_all[x_all_idx]
        x_sampled = torch.LongTensor(x_sampled).to(device)
        sub_obs = torch.Tensor(obs[x_all_idx]).unsqueeze(-1).to(device)
        sub_entire_y = torch.Tensor(y_entire[x_all_idx]).unsqueeze(-1).to(device)

        prop_pred, _, __ = model.propensity_model(sub_x)
        inv_prop = 1/torch.clip(nn.Sigmoid()(prop_pred), gamma, 1)
        pred, _, __ = model.prediction_model(sub_x)
        pred = nn.Sigmoid()(pred)
        ips_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
        # ctr loss

        prop_pred_all, _, __ =  model.propensity_model(x_sampled)
        inv_prop_all = 1/torch.clip(nn.Sigmoid()(prop_pred_all), gamma, 1)
        prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)

        pred_all, _, __ = model.prediction_model(x_sampled)
        pred_all_loss = F.binary_cross_entropy(1/inv_prop_all * nn.Sigmoid()(pred_all), sub_entire_y)

        ones_all = torch.ones(len(inv_prop_all)).unsqueeze(-1).to(device)
        w_all = torch.divide(sub_obs,1/inv_prop_all) - torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
        bmse_loss = (torch.mean(w_all * pred_all))**2
                
        total_loss = alpha * prop_loss + beta * pred_all_loss + ips_loss +  eta * bmse_loss

        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    # loss_dict: dict = {
    #     'epoch_rec_loss': float(epoch_rec_loss.item()),
    #     'epoch_total_loss': float(epoch_total_loss.item()),
    # }

    # wandb_var.log(loss_dict)

    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = ndcg_func(model.prediction_model, x_test, y_test, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        # wandb_var.log(ndcg_dict)

        recall_res = recall_func(model.prediction_model, x_test, y_test, device, top_k_list)
        recall_dict: dict = {}
        for top_k in top_k_list:
            recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])
        # wandb_var.log(recall_dict)

        ap_res = ap_func(model.prediction_model, x_test, y_test, device, top_k_list)
        ap_dict: dict = {}
        for top_k in top_k_list:
            ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])
        # wandb_var.log(ap_dict)

# wandb.finish()
print(f"NDCG: {ndcg_dict}")
print(f"Recall: {recall_dict}")
print(f"AP: {ap_dict}")
