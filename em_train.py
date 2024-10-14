#%%
WANDB_TRACKING = 0
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import sparse
from datetime import datetime

from module.model import MF
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import binarize

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb
finally: 
    WANDB_TRACKING = 1


def generate_total_sample(num_users, num_items):
    sample = []
    for i in range(num_users):
        sample.extend([[i,j] for j in range(num_items)])

    return np.array(sample)


def pairwise_loss(pos_pred, neg_pred):
    return torch.nn.Sigmoid()(pos_pred - neg_pred).mean()


class Posterior(nn.Module):
    def __init__(self, preference_model, exposure_model):
        super(Posterior, self).__init__()
        self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)

        self.preference_model = preference_model
        for param in self.preference_model.parameters():
            param.requires_grad = False

        self.exposure_model = exposure_model
        for param in self.exposure_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        preference_pred, _, __ = self.preference_model(x)
        preference_pred = nn.Sigmoid()(preference_pred)
        exposure_pred, _, __ = self.exposure_model(x)
        exposure_odds = torch.exp(exposure_pred + self.gamma)
        posterior = preference_pred / (1 + exposure_odds * (1 - preference_pred))
        return posterior


def q_objective_fn(x, y_o, posterior):
    preference_logit, _, __ = posterior.preference_model(x)
    preference_prob = nn.Sigmoid()(preference_logit)

    exposure_logit, _, __ = posterior.exposure_model(x)
    exposure_prob = nn.Sigmoid()(exposure_logit + posterior.gamma)

    q_y1_o1_mask = (y_o.sum(-1) == 2).float().unsqueeze(-1)
    q_y0_o1_mask = (y_o.sum(-1) == 1).float().unsqueeze(-1)
    q_y0_o0_mask = (y_o.sum(-1) == 0).float().unsqueeze(-1)

    q_y1_o1 = q_y1_o1_fn(x, preference_prob, exposure_prob) * q_y1_o1_mask
    q_y0_o1 = q_y0_o1_fn(x, preference_prob, exposure_prob) * q_y0_o1_mask
    q_y0_o0 = q_y0_o0_fn(x, preference_prob, exposure_prob, posterior) * q_y0_o0_mask

    return (q_y1_o1 + q_y0_o1 + q_y0_o0).mean()


def q_y1_o1_fn(x, preference_prob, exposure_prob):
    return torch.log(preference_prob * exposure_prob)


def q_y0_o1_fn(x, preference_prob, exposure_prob):
    return torch.log((1 - preference_prob) * exposure_prob)


def q_y0_o0_fn(x, preference_prob, exposure_prob, posterior):
    posterior_r1 = posterior(x)
    posterior_r0 = 1 - posterior_r1

    q_y0_o0_r1 = posterior_r1 * torch.log(preference_prob * (1 - exposure_prob))
    q_y0_o0_r0 = posterior_r0 * torch.log((1 - preference_prob) * (1 - exposure_prob))

    return q_y0_o0_r1 + q_y0_o0_r0



#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--exposure_neg_size", type=int, default=1)

parser.add_argument("--em-lr", type=float, default=1e-2)
parser.add_argument("--em-batch-size", type=int, default=2048)
parser.add_argument("--em-num-epochs", type=int, default=1000)

parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset-name", type=str, default="ml-1m") # [coat, kuairec, yahoo_r3, ml-1m]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

exposure_neg_size = args.exposure_neg_size

em_lr = args.em_lr
em_batch_size = args.em_batch_size
em_num_epochs = args.em_num_epochs

random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
save_dir = f"./weights/expt_{expt_num}"
os.makedirs(f"{save_dir}", exist_ok=True)

config = vars(args)
config["device"] = device
config["expt_num"] = expt_num
config["save_dir"] = save_dir

if WANDB_TRACKING:
    wandb_var = wandb.init(project="drs", config=config)
    wandb.run.name = f"main_{expt_num}"


#%% OBSERVED DATA LOADER
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

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

if dataset_name != "kuairec":
    y_train = binarize(y_train)
    y_test = binarize(y_test)

num_sample = len(x_train)
total_batch = num_sample // em_batch_size

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()

print("===>Load from {} data set<===".format(dataset_name))
print("[train] num data:", x_train.shape[0])
print("[test]  num data:", x_test.shape[0])
print("# user: {}, # item: {}".format(num_users, num_items))
print("# prefer: {}, # not prefer: {}".format(y_train.sum(), num_sample - y_train.sum()))


# %% UNOBSERVED DATA LOADER
np.random.seed(random_seed)
torch.manual_seed(random_seed)

exposure_matrix = sparse.lil_matrix((num_users, num_items))
for (u, i) in x_train:
    exposure_matrix[int(u)-1, int(i)-1] = 1
exposure_train = sparse.csr_matrix(exposure_matrix)

x_all = generate_total_sample(num_users, num_items)
o_all = np.array([exposure_matrix[u-1,i-1] for (u,i) in x_all])
zero_indices = np.argwhere(o_all == 0).squeeze(-1)
unexposed_pairs_all = x_all[zero_indices, :]

num_neg_sample = num_sample * exposure_neg_size
num_neg_per_item = num_neg_sample // num_items + 1

unexposed_pairs = []
for i in range(num_items):
    unexposed_users = unexposed_pairs_all[unexposed_pairs_all[:, 1] == i]
    sampling_indices = np.random.choice(np.arange(len(unexposed_users)), num_neg_per_item)
    unexposed_pairs.append(unexposed_users[sampling_indices])
unexposed_pairs = np.concatenate(unexposed_pairs, 0) + 1

del exposure_matrix, exposure_train, x_all, o_all, zero_indices, unexposed_pairs_all


#%% LOAD MODEL 
np.random.seed(random_seed)
torch.manual_seed(random_seed)

preference_model = torch.load("/Users/wonhyung64/Github/DRS/preference.pth")
preference_model = preference_model.to(device)

exposure_model = torch.load("/Users/wonhyung64/Github/DRS/exposure_pairwise_302409e.pth")
exposure_model = exposure_model.to(device)

posterior = Posterior(preference_model, exposure_model)
posterior = posterior.to(device)

optimizer = torch.optim.Adam(posterior.parameters(), lr=em_lr)


#%% EM-algorithm
np.random.seed(random_seed)
torch.manual_seed(random_seed)

num_pos_exposure = len(x_train)
num_neg_exposure = len(unexposed_pairs)
total_batch = num_pos_exposure // (em_batch_size//2)
pos_exposure_idx = np.arange(num_pos_exposure)
neg_exposure_idx = np.arange(num_neg_exposure)

for epoch in range(1, em_num_epochs+1):
    np.random.shuffle(pos_exposure_idx)
    np.random.shuffle(neg_exposure_idx)

    posterior.train()
    epoch_q_objective = 0.

    for idx in range(total_batch):
        pos_idx = pos_exposure_idx[(em_batch_size//2)*idx : (idx+1)*(em_batch_size//2)]
        exposed_x = x_train[pos_idx]
        exposed_x = torch.LongTensor(exposed_x - 1).to(device)
        exposed_y_ = y_train[pos_idx]
        exposed_y_ = np.array([exposed_y_, np.ones_like(exposed_y_)]).T
        exposed_y = torch.LongTensor(exposed_y_).to(device)

        neg_idx = neg_exposure_idx[(em_batch_size//2)*idx : (idx+1)*(em_batch_size//2)]
        unexposed_x = unexposed_pairs[neg_idx]
        unexposed_x = torch.LongTensor(unexposed_x - 1).to(device)
        unexposed_y = np.zeros_like(exposed_y_)
        unexposed_y = torch.LongTensor(unexposed_y).to(device)

        sub_x = torch.cat([exposed_x, unexposed_x])
        sub_y_o = torch.cat([exposed_y, unexposed_y])

        q_objective = -q_objective_fn(sub_x, sub_y_o, posterior)

        optimizer.zero_grad()
        q_objective.backward()
        optimizer.step()
        print(posterior.gamma.item())

        epoch_q_objective += q_objective

    print(f"[Epoch {epoch:>4d} Train Loss] Q-objective: {epoch_q_objective.item():.4f}")

    if WANDB_TRACKING:
        loss_dict: dict = {
            'epoch_q_objective': float(epoch_q_objective.item()),
        }
        wandb_var.log(loss_dict)

    if epoch % evaluate_interval == 0:
        posterior.eval()

        ndcg_res = ndcg_func(posterior, x_test, y_test, device, top_k_list)
        recall_res = recall_func(posterior, x_test, y_test, device, top_k_list)
        ap_res = ap_func(posterior, x_test, y_test, device, top_k_list)

        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

        recall_dict: dict = {}
        for top_k in top_k_list:
            recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

        ap_dict: dict = {}
        for top_k in top_k_list:
            ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

        print(f"NDCG: {ndcg_dict}")
        print(f"Recall: {recall_dict}")
        print(f"AP: {ap_dict}")

        if WANDB_TRACKING:
            wandb_var.log(ndcg_dict)
            wandb_var.log(recall_dict)
            wandb_var.log(ap_dict)

if WANDB_TRACKING:
    wandb.finish()

# %%
