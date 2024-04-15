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
parser.add_argument("--ipw-sampling", type=bool, default=True)
parser.add_argument("--ipw-erm", type=bool, default=True)


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

"""USER PAIRS"""
train_user_indices = x_train.copy()[:, 0] - 1
if ipw_sampling:
    pos_user_samples_ = np.load("./assets/ipw_pos_user_topk.npy")
    neg_user_samples_ = np.load("./assets/ipw_neg_user_samples.npy")
else:
    pos_user_samples_ = np.load("./assets/pos_user_topk.npy")
    neg_user_samples_ = np.load("./assets/neg_user_samples.npy")

pos_user_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_user_samples_))
pos_user_samples_ = np.array([pos_user_samples_[:,:pos_topk][i, pos_user_indices[i]] for i in range(len(pos_user_indices))])

pos_user_samples = pos_user_samples_[train_user_indices]
neg_user_samples = neg_user_samples_[train_user_indices]
user_pos_neg = np.stack([pos_user_samples, neg_user_samples], axis=-1)

"""ITEM PAIRS"""
train_item_indices = x_train.copy()[:, 1] - 1
if ipw_sampling:
    pos_item_samples_ = np.load("./assets/ipw_pos_item_topk.npy")
    neg_item_samples_ = np.load("./assets/ipw_neg_item_samples.npy")
else:
    pos_item_samples_ = np.load("./assets/pos_item_topk.npy")
    neg_item_samples_ = np.load("./assets/neg_item_samples.npy")

pos_item_indices = np.random.randint(low=0, high=pos_topk, size=len(pos_item_samples_))
pos_item_samples_ = np.array([pos_item_samples_[:,:pos_topk][i, pos_item_indices[i]] for i in range(len(pos_item_indices))])

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
total_batch = num_sample // batch_size


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

        pos_x = np.stack([user_pos_neg[selected_idx,0], item_pos_neg[selected_idx,0]], axis=-1)
        pos_x = torch.LongTensor(pos_x - 1).to(device)
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

wandb.finish()
