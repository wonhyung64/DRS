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

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def weight_fcn(similarity, scale_param):
    return torch.exp(similarity) / torch.relu(scale_param + 1e-9)


def estimate_ips_bayes(x, y, y_ips=None):
    if y_ips is None:
        one_over_zl = np.ones(len(y))
    else:
        prob_y1 = y_ips.sum() / len(y_ips)
        prob_y0 = 1 - prob_y1
        prob_o1 = len(x) / (x[:,0].max() * x[:,1].max())
        prob_y1_given_o1 = y.sum() / len(y)
        prob_y0_given_o1 = 1 - prob_y1_given_o1

        propensity = np.zeros(len(y))

        propensity[y == 0] = (prob_y0_given_o1 * prob_o1) / prob_y0
        propensity[y == 1] = (prob_y1_given_o1 * prob_o1) / prob_y1
        one_over_zl = 1 / propensity

    one_over_zl = torch.Tensor(one_over_zl)

    return one_over_zl

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


#%% WandB
wandb_var = wandb.init(
    project="drs",
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
        "balance_param": balance_param,
        "contrast_pair": contrast_pair,
        "dataset_name": dataset_name,
        "base_model": base_model,
        "sim_measure": sim_measure,
        "temperatrue": temperature,
    }
)
wandb.run.name = f"ours3_{expt_num}"


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

"""USER PAIRS"""
train_user_indices = x_train.copy()[:, 0] - 1
top1_sim_user = user_user_sim.argmax(-1)+1
train_pos_user = top1_sim_user[train_user_indices]

"""ITEM PAIRS"""
train_item_indices = x_train.copy()[:, 1] - 1
top1_sim_item = item_item_sim.argmax(-1)+1
train_pos_item = top1_sim_item[train_item_indices]


#%% TRAIN INITIAILIZE
if base_model == "ncf":
    model = NCF(num_users, num_items, embedding_k)
elif base_model == "mf":
    model = MF(num_users, num_items, embedding_k)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = lambda x, y, z: F.binary_cross_entropy(x, y, z)
loss__ = torch.nn.BCELoss()

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

one_over_zl = estimate_ips_bayes(x_train, y_train, y_ips)

# TRAIN
for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    epoch_total_loss = 0.
    epoch_rec_loss = 0.
    epoch_cl_loss = 0.

    epoch_user_sim_loss = None
    epoch_item_sim_loss = None

    if contrast_pair in ["user", "both"]:
        epoch_user_sim_loss = 0.
    if contrast_pair in ["item", "both"]:
        epoch_item_sim_loss = 0.


    for idx in range(total_batch):
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        org_x = torch.LongTensor(sub_x - 1).to(device)

        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, anchor_user_embed, anchor_item_embed = model(org_x)
        pred = torch.nn.Sigmoid()(pred)

        inv_prop = one_over_zl[selected_idx].unsqueeze(-1).to(device)

        rec_loss = loss_fcn(pred, sub_y, inv_prop)

        epoch_rec_loss += rec_loss

        sub_pos_user = train_pos_user[selected_idx]
        sub_pos_item = train_pos_item[selected_idx]
        pos_x = torch.stack([sub_pos_user-1, sub_pos_item-1], -1).to(device)

        _, pos_user_embed, pos_item_embed = model(pos_x)


        if contrast_pair == "user":
            user_sim_loss = contrastive_loss(anchor_user_embed, pos_user_embed, temperature) * balance_param
            cl_loss = user_sim_loss 
            epoch_user_sim_loss += user_sim_loss
            epoch_cl_loss += cl_loss
            
        elif contrast_pair == "item":
            item_sim_loss = contrastive_loss(anchor_item_embed, pos_item_embed, temperature) * balance_param
            cl_loss = item_sim_loss
            epoch_item_sim_loss += item_sim_loss
            epoch_cl_loss += cl_loss

        elif contrast_pair == "both":
            user_sim_loss = contrastive_loss(anchor_user_embed, pos_user_embed, temperature) * balance_param
            item_sim_loss = contrastive_loss(anchor_item_embed, pos_item_embed, temperature) * balance_param
            cl_loss = (user_sim_loss + item_sim_loss)
            epoch_user_sim_loss += user_sim_loss
            epoch_item_sim_loss += item_sim_loss
            epoch_cl_loss += cl_loss

        else: 
            raise ValueError("Unknown contrastive learning pair!")

        total_loss = rec_loss + cl_loss
        epoch_total_loss += total_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_rec_loss': float(epoch_rec_loss.item()),
        'epoch_cl_loss': float(epoch_cl_loss.item()),
        'epoch_total_loss': float(epoch_total_loss.item()),
    }

    if epoch_user_sim_loss != None:
        loss_dict["epoch_user_sim_loss"] = float(epoch_user_sim_loss.item())
    if epoch_item_sim_loss != None:
        loss_dict["epoch_item_sim_loss"] = float(epoch_item_sim_loss.item())

    wandb_var.log(loss_dict)


    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = ndcg_func(model, x_test, y_test, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        wandb_var.log(ndcg_dict)

        recall_res = recall_func(model, x_test, y_test, device, top_k_list)
        recall_dict: dict = {}
        for top_k in top_k_list:
            recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])
        wandb_var.log(recall_dict)

        ap_res = ap_func(model, x_test, y_test, device, top_k_list)
        ap_dict: dict = {}
        for top_k in top_k_list:
            ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])
        wandb_var.log(ap_dict)

wandb.finish()
print(f"NDCG: {ndcg_dict}")
print(f"Recall: {recall_dict}")
print(f"AP: {ap_dict}")

# %%