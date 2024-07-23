#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from utils import binarize

import torch.nn as nn
from datetime import datetime

from model import InvPrefImplicit, _init_eps
from metric import ndcg_func, recall_func, ap_func
from clustering import em_clustering

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def stat_envs(envs, envs_num, scores_tensor):
    result: dict = dict()
    class_rate_np: np.array = np.zeros(envs_num)
    for env in range(envs_num):
        cnt: int = int(torch.sum(envs == env))
        result[env] = cnt
        class_rate_np[env] = min(cnt + 1, scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / scores_tensor.shape[0]
        class_weights = torch.Tensor(class_rate_np).to(device)
        sample_weights = class_weights[envs]

    return class_weights, sample_weights, result


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=40)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../../data")
parser.add_argument("--dataset-name", type=str, default="yahoo_r3")
parser.add_argument("--env-num", type=int, default=2)
parser.add_argument("--cluster-interval", type=int, default=5)
parser.add_argument("--invariant-coe", type=float, default=3.351991776096847)
parser.add_argument("--env-aware-coe", type=float, default=9.988658447411407)
parser.add_argument("--env-coe", type=float, default=9.06447753571379)
parser.add_argument("--L2-coe", type=float, default=3.1351402017943117)
parser.add_argument("--L1-coe", type=float, default=0.4935216278026648)
parser.add_argument("--alpha", type=float, default=1.9053711444718746)
parser.add_argument("--reg-only-embed", type=bool, default=True)
parser.add_argument("--reg-env-embed", type=bool, default=False)
parser.add_argument("--use-class-re-weight", type=bool, default=True)
parser.add_argument("--use-recommend-re-weight", type=bool, default=False)
parser.add_argument("--test-begin-epoch", type=int, default=0)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

# DATA CONFIG
data_dir = args.data_dir
dataset_name = args.dataset_name

# MODEL CONFIG
env_num = args.env_num
embedding_k = args.embedding_k
reg_only_embed = args.reg_only_embed
reg_env_embed = args.reg_env_embed

# TRAIN CONFIG
batch_size = args.batch_size
num_epochs = args.num_epochs
cluster_interval = args.cluster_interval
evaluate_interval = args.evaluate_interval
lr = args.lr
invariant_coe = args.invariant_coe
env_aware_coe = args.env_aware_coe
env_coe = args.env_coe
L2_coe = args.L2_coe
L1_coe = args.L1_coe
alpha = args.alpha
use_class_re_weight = args.use_class_re_weight
use_recommend_re_weight = args.use_recommend_re_weight
test_begin_epoch = args.test_begin_epoch

# EVALUATE_CONFIG
top_k_list = args.top_k_list

# EXPERIMENT CONFIG
random_seed = args.random_seed

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
"""WandB"""
wandb_var = wandb.init(
    project="drs",
    config={
        "device" : device,
        "env_num" : env_num,
        "embedding_k" : embedding_k,
        "reg_only_embed" : reg_only_embed,
        "reg_env_embed" : reg_env_embed,
        "batch_size" : batch_size,
        "num_epochs" : num_epochs,
        "cluster_interval" : cluster_interval,
        "evaluate_interval" : evaluate_interval,
        "lr" : lr,
        "invariant_coe" : invariant_coe,
        "env_aware_coe" : env_aware_coe,
        "env_coe" : env_coe,
        "L2_coe" : L2_coe,
        "L1_coe" : L1_coe,
        "alpha" : alpha,
        "use_class_re_weight" : use_class_re_weight,
        "use_recommend_re_weight" : use_recommend_re_weight,
        "top_k_list" : top_k_list,
        "random_seed" : random_seed,
        "data_dir" : data_dir,
        "dataset_name" : dataset_name,
    }
)
wandb.run.name = f"invpref_{expt_num}"


#%% DataLoader
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
model = InvPrefImplicit(
    user_num=num_users,
    item_num=num_items,
    env_num=env_num,
    embedding_k=embedding_k,
    reg_only_embed=reg_only_embed,
    reg_env_embed=reg_env_embed
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

recommend_loss_type = nn.BCELoss
if use_recommend_re_weight:
    recommend_loss_fcn = recommend_loss_type(reduction='none')
else:
    recommend_loss_fcn = recommend_loss_type()

env_loss_type = nn.NLLLoss
if use_class_re_weight:
    env_loss_fcn = env_loss_type(reduction='none')
else:
    env_loss_fcn = env_loss_type()

cluster_distance_fcn = nn.BCELoss(reduction='none')

users_tensor = torch.LongTensor(x_train[:,0]-1).to(device)
items_tensor = torch.LongTensor(x_train[:,1]-1).to(device)
scores_tensor = torch.LongTensor(y_train).to(device)
envs = torch.LongTensor(np.random.randint(0, env_num, num_sample)).to(device)

if alpha is None:
    alpha = 0.
    update_alpha = True
else:
    alpha = alpha
    update_alpha = False

sample_weights: torch.Tensor = torch.Tensor(np.zeros(num_sample)).to(device)
class_weights: torch.Tensor = torch.Tensor(np.zeros(env_num)).to(device)
eps_random_tensor: torch.Tensor = _init_eps(env_num).to(device)

const_env_tensor_list: list = []
for env in range(env_num):
    envs_tensor: torch.Tensor = torch.LongTensor(np.full(num_sample, env, dtype=int))
    envs_tensor = envs_tensor.to(device)
    const_env_tensor_list.append(envs_tensor)

class_weights, sample_weights, result = stat_envs(envs, env_num, scores_tensor)

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    epoch_total_loss = 0.
    epoch_env_loss = 0.
    epoch_invariant_loss = 0.
    epoch_env_aware_loss = 0.
    epoch_L2_reg = 0.
    epoch_L1_reg = 0.

    for idx in range(total_batch):
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        batch_users_tensor = sub_x[:,0]
        batch_items_tensor = sub_x[:,1]
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
        sub_envs = envs[selected_idx]
        sub_sample_weights = sample_weights[selected_idx]

        if update_alpha:
            p = float(idx + (epoch) * total_batch) / float((epoch) * total_batch)
            alpha = 2. / (1. + np.exp(-10. * p)) - 1.

        invariant_score, env_aware_score, env_outputs = model(
            batch_users_tensor, batch_items_tensor, sub_envs, alpha
        )

        envs_loss: torch.Tensor = env_loss_fcn(env_outputs, sub_envs)
        if use_class_re_weight:
            envs_loss = torch.mean(envs_loss * sub_sample_weights)
        envs_loss *= env_coe
        epoch_env_loss += envs_loss

        invariant_loss: torch.Tensor = recommend_loss_fcn(invariant_score, sub_y.squeeze(-1))
        env_aware_loss: torch.Tensor = recommend_loss_fcn(env_aware_score, sub_y.squeeze(-1))
        if use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * sub_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * sub_sample_weights)
        invariant_loss *= invariant_coe
        env_aware_loss *= env_aware_coe
        epoch_invariant_loss += invariant_loss
        epoch_env_aware_loss += env_aware_loss

        L2_reg: torch.Tensor = model.get_L2_reg(batch_users_tensor, batch_items_tensor, sub_envs) * L2_coe
        L1_reg: torch.Tensor = model.get_L1_reg(batch_users_tensor, batch_items_tensor, sub_envs) * L1_coe
        epoch_L2_reg += L2_reg
        epoch_L1_reg += L1_reg


        total_loss: torch.Tensor = invariant_loss + env_aware_loss + envs_loss + L2_reg + L1_reg
        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_rec_loss': float(epoch_env_loss.item()),
        'epoch_rec_loss': float(epoch_invariant_loss.item()),
        'epoch_rec_loss': float(epoch_env_aware_loss.item()),
        'epoch_rec_loss': float(epoch_L2_reg.item()),
        'epoch_rec_loss': float(epoch_L1_reg.item()),
        'epoch_total_loss': float(epoch_total_loss.item()),
    }
    wandb_var.log(loss_dict)

    if (epoch-1 % cluster_interval) == 0:
        print("clustering")
        envs, diff_num = em_clustering(
            batch_size,
            env_num,
            num_sample,
            total_batch,
            device,
            model,
            x_train,
            y_train,
            envs,
            const_env_tensor_list,
            cluster_distance_fcn,
            )
        class_weights, sample_weights, result = stat_envs(envs, env_num, scores_tensor)
        wandb_var.log({"env_diff_num": diff_num})

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
