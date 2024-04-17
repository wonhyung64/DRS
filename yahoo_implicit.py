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


def compute_sim_matrix(pos_interactions, k=5):
    pref_user_sim_ = torch.matmul(pos_interactions, pos_interactions.T).cpu().numpy()
    pref_user_sim = pref_user_sim_ * (np.ones_like(pref_user_sim_) - np.identity(num_users)*2)
    pref_user_topk = torch.topk(torch.tensor(pref_user_sim), k).indices + 1

    pref_item_sim_ = torch.matmul(pos_interactions.T, pos_interactions).cpu().numpy()
    pref_item_sim = pref_item_sim_ * (np.ones_like(pref_item_sim_) - np.identity(num_items)*2)
    pref_item_topk = torch.topk(torch.tensor(pref_item_sim), k).indices + 1

    return pref_user_topk, pref_item_topk


def hard_contrastive_loss(anchor_embed, aug_embed, scale=1.):
    batch_size = anchor_embed.shape[0]
    device = anchor_embed.device
    anchor_embed = F.normalize(anchor_embed, p=2, dim=1)
    aug_embed = F.normalize(aug_embed, p=2, dim=1)
    simlarity = (anchor_embed.unsqueeze(1) * aug_embed).sum(-1) / scale
    target = torch.LongTensor(np.zeros(batch_size)).to(device)

    return torch.nn.functional.cross_entropy(simlarity, target)


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.)
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
parser.add_argument("--pos-topk", type=int, default=1)
parser.add_argument("--neg-topk", type=int, default=5)
parser.add_argument("--ipw-sampling", type=bool, default=True)
parser.add_argument("--ipw-erm", type=bool, default=False)
parser.add_argument("--pref-update-interval", type=int, default=25)


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
pos_topk = args.pos_topk
ipw_sampling = args.ipw_sampling
ipw_erm = args.ipw_erm
pref_update_interval = args.pref_update_interval
neg_topk = args.neg_topk


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
        "temperature": temperature,
        "balance_param": balance_param,
        "contrast_pair": contrast_pair,
        "pos_topk": pos_topk,
        "ipw_sampling": ipw_sampling,
        "ipw_erm": ipw_erm,
        "pref_update_interval": pref_update_interval,
        "neg_topk": neg_topk,
    }
)
wandb.run.name = f"ours_{expt_num}"


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

y_train = binarize(y_train)
y_test = binarize(y_test)

num_sample = len(x_train)
total_batch = num_sample // batch_size

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print("# user: {}, # item: {}".format(num_users, num_items))


"""INTERACTION MATRIX"""
pos_interactions = torch.tensor(np.load("./assets/pos_interactions.npy")).to(device)

if ipw_sampling:
    popularity = pos_interactions.sum(dim=0) / num_users
    init_pos_interactions = pos_interactions / popularity
else:
    init_pos_interactions = pos_interactions

pos_user_topk_, pos_item_topk_ = compute_sim_matrix(init_pos_interactions)


"""USER PAIRS"""
train_user_indices = x_train.copy()[:, 0] - 1
pos_user_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_user_topk_))
pos_user_top1_ = np.array([pos_user_topk_[:,:pos_topk][i, pos_user_indices[i]] for i in range(len(pos_user_indices))])

"""ITEM PAIRS"""
train_item_indices = x_train.copy()[:, 1] - 1
pos_item_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_item_topk_))
pos_item_top1_ = np.array([pos_item_topk_[:,:pos_topk][i, pos_item_indices[i]] for i in range(len(pos_item_indices))])

"""POSITIVE"""
pos_user_top1 = pos_user_top1_[train_user_indices]
pos_item_top1 = pos_item_top1_[train_item_indices]

"""HARD NEGATIVES"""
if neg_topk:
    neg_interactions = torch.tensor(np.load("./assets/neg_interactions.npy")).to(device)
    neg_user_topk_, neg_item_topk_ = compute_sim_matrix(neg_interactions, k=neg_topk)
    neg_user_topk = neg_user_topk_[train_user_indices]
    neg_item_topk = neg_item_topk_[train_item_indices]


#%% TRAIN INITIAILIZE
model = NCF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()


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

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        org_x = torch.LongTensor(sub_x - 1).to(device)

        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, anchor_user_embed, anchor_item_embed = model(org_x)
        pred = torch.nn.Sigmoid()(pred)

        if ipw_erm:
            ppscore = torch.clip(pred, min=0.1, max=1.0)
            true = sub_y / ppscore
            rec_loss = torch.mean(true * torch.square(1 - pred) + (1 - true) * torch.square(pred))
        else:
            rec_loss = loss_fcn(pred, sub_y)
        epoch_rec_loss += rec_loss

        pos_x = np.stack([pos_user_top1[selected_idx], pos_item_top1[selected_idx]], axis=-1)
        pos_x = torch.LongTensor(pos_x - 1).to(device)
        _, pos_user_embed, pos_item_embed = model(pos_x)

        if neg_topk:
            neg_x_user = neg_user_topk[selected_idx].flatten()
            neg_x_item = neg_item_topk[selected_idx].flatten()
            neg_x = np.stack([neg_x_user, neg_x_item], axis=-1)
            neg_x = torch.LongTensor(neg_x - 1).to(device)
            _, neg_user_embed, neg_item_embed = model(neg_x)

            aug_user_embed = torch.cat([
                pos_user_embed.reshape(batch_size, 1, embedding_k),
                neg_user_embed.reshape(batch_size, neg_topk, embedding_k),
            ], dim=1)

            aug_item_embed = torch.cat([
                pos_item_embed.reshape(batch_size, 1, embedding_k),
                neg_item_embed.reshape(batch_size, neg_topk, embedding_k),
            ], dim=1)

        if contrast_pair == "user":
            if neg_topk:
                user_sim_loss = hard_contrastive_loss(anchor_user_embed, aug_user_embed, temperature) * balance_param
            else:
                user_sim_loss = contrastive_loss(anchor_user_embed, pos_user_embed, temperature) * balance_param
            cl_loss = user_sim_loss
            epoch_user_sim_loss += user_sim_loss
            epoch_cl_loss += cl_loss
            
        elif contrast_pair == "item":
            if neg_topk:
                item_sim_loss = hard_contrastive_loss(anchor_item_embed, aug_item_embed, temperature) * balance_param
            else:
                item_sim_loss = contrastive_loss(anchor_item_embed, pos_item_embed, temperature) * balance_param
            cl_loss = item_sim_loss
            epoch_item_sim_loss += item_sim_loss
            epoch_cl_loss += cl_loss

        elif contrast_pair == "both":
            if neg_topk:
                user_sim_loss = hard_contrastive_loss(anchor_user_embed, aug_user_embed, temperature) * balance_param
                item_sim_loss = hard_contrastive_loss(anchor_item_embed, aug_item_embed, temperature) * balance_param
            else:
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

    if epoch % pref_update_interval == 0:

        propensity_score = []
        for u in range(num_users):
            with torch.no_grad():
                u_items = torch.LongTensor(np.array([np.ones(num_items)*u, np.arange(num_items)])).T.to(device)
                propensity_u, _, __ = model(u_items)
                propensity_u = torch.nn.Sigmoid()(propensity_u)
                propensity_score.append(propensity_u)
        propensity_score = torch.concat(propensity_score, dim=-1).T

        weighted_pos_interactions = pos_interactions / propensity_score
        pos_user_topk_, pos_item_topk_ = compute_sim_matrix(weighted_pos_interactions)

        """USER PAIRS"""
        pos_user_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_user_topk_))
        pos_user_top1_ = np.array([pos_user_topk_[:,:pos_topk][i, pos_user_indices[i]] for i in range(len(pos_user_indices))])

        """ITEM PAIRS"""
        pos_item_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_item_topk_))
        pos_item_top1_ = np.array([pos_item_topk_[:,:pos_topk][i, pos_item_indices[i]] for i in range(len(pos_item_indices))])

        """POSITIVE"""
        pos_user_top1 = pos_user_top1_[train_user_indices]
        pos_item_top1 = pos_item_top1_[train_item_indices]

wandb.finish()

# %%
