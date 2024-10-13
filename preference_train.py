WANDB_TRACKING = 0
#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

from module.model import MF
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import binarize

# try:
#     import wandb
# except: 
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
#     import wandb
# finally: 
#     WANDB_TRACKING = 1


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--preference-factor-dim", type=int, default=4)
parser.add_argument("--preference-lr", type=float, default=1e-2)
parser.add_argument("--preference-weight-decay", type=float, default=1e-4)
parser.add_argument("--preference-batch-size", type=int, default=2048)

parser.add_argument("--exposure-factor-dim", type=int, default=4)
parser.add_argument("--exposure-lr", type=float, default=1e-2)
parser.add_argument("--exposure-weight-decay", type=float, default=1e-4)
parser.add_argument("--exposure-batch-size", type=int, default=2048)
parser.add_argument("--exposure_neg_size", type=int, default=1)

parser.add_argument("--em-lr", type=float, default=1e-2)
parser.add_argument("--em-batch-size", type=int, default=2048)

parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../../data")
parser.add_argument("--dataset-name", type=str, default="coat") # [coat, kuairec, yahoo]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


preference_factor_dim = args.preference_factor_dim
preference_lr = args.preference_lr
preference_weight_decay = args.preference_weight_decay
preference_batch_size = args.preference_batch_size

exposure_factor_dim = args.exposure_factor_dim
exposure_lr = args.exposure_lr
exposure_weight_decay = args.exposure_weight_decay
exposure_batch_size = args.exposure_batch_size
exposure_neg_size = args.exposure_neg_size

em_lr = args.em_lr
em_batch_size = args.em_batch_size

num_epochs = args.num_epochs
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

config = vars(args)
config["device"] = device
config["expt_num"] = expt_num
config["save_dir"] = save_dir

os.makedirs(f"{save_dir}", exist_ok=True)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


if WANDB_TRACKING:
    wandb_var = wandb.init(project="drs", config=config)
    wandb.run.name = f"main_{expt_num}"


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

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

if dataset_name != "kuairec":
    y_train = binarize(y_train)
    y_test = binarize(y_test)

num_sample = len(x_train)
total_batch = num_sample // preference_batch_size

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()

print("===>Load from {} data set<===".format(dataset_name))
print("[train] num data:", x_train.shape[0])
print("[test]  num data:", x_test.shape[0])
print("# user: {}, # item: {}".format(num_users, num_items))


#%% TRAIN
preference_model = MF(num_users, num_items, preference_factor_dim)
preference_model = preference_model.to(device)
optimizer = torch.optim.Adam(preference_model.parameters(), lr=preference_lr, weight_decay=preference_weight_decay)
loss_fcn = torch.nn.BCELoss()

all_idx = np.arange(num_sample)
for epoch in range(1, num_epochs+1):
    np.random.shuffle(all_idx)
    preference_model.train()
    epoch_preference_loss = 0.

    for idx in range(total_batch):
        selected_idx = all_idx[preference_batch_size*idx : (idx+1)*preference_batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = preference_model(sub_x)

        preference_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)
        epoch_preference_loss += preference_loss

        optimizer.zero_grad()
        preference_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] preference: {epoch_preference_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_preference_loss': float(epoch_preference_loss.item()),
    }

    if WANDB_TRACKING:
        wandb_var.log(loss_dict)

#     if epoch % evaluate_interval == 0:
#         model.eval()

#         ndcg_res = ndcg_func(model, x_test, y_test, device, top_k_list)
#         ndcg_dict: dict = {}
#         for top_k in top_k_list:
#             ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

#         recall_res = recall_func(model, x_test, y_test, device, top_k_list)
#         recall_dict: dict = {}
#         for top_k in top_k_list:
#             recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

#         ap_res = ap_func(model, x_test, y_test, device, top_k_list)
#         ap_dict: dict = {}
#         for top_k in top_k_list:
#             ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

#         if WANDB_TRACKING:
#             wandb_var.log(ndcg_dict)
#             wandb_var.log(recall_dict)
#             wandb_var.log(ap_dict)

# print(f"NDCG: {ndcg_dict}")
# print(f"Recall: {recall_dict}")
# print(f"AP: {ap_dict}")

# if WANDB_TRACKING:
#     wandb.finish()

# %%
from scipy import sparse

def generate_total_sample(num_users, num_items):
    sample = []
    for i in range(num_users):
        sample.extend([[i,j] for j in range(num_items)])

    return np.array(sample)


def pairwise_loss(pos_pred, neg_pred):
    return torch.nn.Sigmoid()(pos_pred - neg_pred).mean()


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



#%%
exposure_model = MF(num_users, num_items, exposure_factor_dim)
exposure_model = exposure_model.to(device)
optimizer = torch.optim.Adam(exposure_model.parameters(), lr=exposure_lr, weight_decay=exposure_weight_decay)

if exposure_neg_size == 1:
    loss_fcn = pairwise_loss
elif exposure_neg_size > 1:
    pass
else:
    raise ValueError("unknown exposure_neg_size")

all_idx = np.arange(num_sample)
# for epoch in range(1, num_epochs+1):
for epoch in range(1, num_epochs*num_neg_sample+1):
    np.random.shuffle(all_idx)
    np.random.shuffle(unexposed_pairs)

    exposure_model.train()
    epoch_exposure_loss = 0.

    for idx in range(total_batch):
        selected_idx = all_idx[exposure_batch_size*idx : (idx+1)*exposure_batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        unexposed_x = unexposed_pairs[selected_idx]

        unexposed_x = unexposed_pairs[selected_idx]
        unexposed_x = torch.LongTensor(unexposed_x - 1).to(device)

        exposed_pred, exposed_user_embed, exposed_item_embed = exposure_model(sub_x)
        unexposed_pred, unexposed_user_embed, exposed_item_embed = exposure_model(unexposed_x)

        exposure_loss = loss_fcn(exposed_pred, unexposed_pred)
        epoch_exposure_loss += exposure_loss

        optimizer.zero_grad()
        exposure_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] exposure: {epoch_exposure_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_exposure_loss': float(epoch_exposure_loss.item()),
    }

    if WANDB_TRACKING:
        wandb_var.log(loss_dict)


#%% EM-algorithm
import torch.nn as nn

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


posterior = Posterior(preference_model, exposure_model)
posterior = posterior.to(device)
optimizer = torch.optim.Adam(posterior.parameters(), lr=em_lr)

posterior.train()

y_o = torch.tensor([[0,1], [0,0], [1,1]]).to(device)
x = sub_x[:3]

q_objective = -q_objective_fn(x, y_o, posterior)
optimizer.zero_grad()
q_objective.backward()
optimizer.step()
print(posterior.gamma)


# %%
