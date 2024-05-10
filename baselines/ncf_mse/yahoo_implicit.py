#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import torch.nn.functional as F

from scipy import sparse
from datetime import datetime

from model import Imputator, DRMse
from metric import ndcg_func
from utils import binarize

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def generate_total_sample(num_users, num_items):
    sample = []
    for i in range(num_users):
        sample.extend([[i,j] for j in range(num_items)])

    return np.array(sample)


def estimate_ips_bayes(x, y, y_ips=None, with_ps=False):
    if y_ips is None:
        one_over_zl = np.ones(len(y))
    else:
        prob_y1 = y_ips.sum() / len(y_ips)
        prob_y0 = 1 - prob_y1
        prob_o1 = len(x) / (x[:,0].max() * x[:,1].max())
        prob_y1_given_o1 = y.sum() / len(y)
        prob_y0_given_o1 = 1 - prob_y1_given_o1

        propensity = np.zeros(len(y))
        propensity_0 = (prob_y0_given_o1 * prob_o1) / prob_y0
        propensity_1 = (prob_y1_given_o1 * prob_o1) / prob_y1

        propensity[y == 0] = propensity_0
        propensity[y == 1] = propensity_1
        one_over_zl = 1 / propensity

    one_over_zl = torch.Tensor(one_over_zl)

    if with_ps:
        return one_over_zl, propensity_0, propensity_1

    return one_over_zl



#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../../data")
parser.add_argument("--dataset-name", type=str, default="yahoo_r3")
parser.add_argument("--tune-lambda", type=float, default=.5)

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
tune_lambda = args.tune_lambda


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
        "tune_lambda" : tune_lambda,
    }
)
wandb.run.name = f"ncf_drjl_{expt_num}"


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


rating_matrix = sparse.lil_matrix((num_users, num_items))
for (u, i, r) in np.concatenate([x_train, np.expand_dims(y_train, -1)], axis=-1):
    if r:
        rating_matrix[int(u)-1, int(i)-1] = r
rating_train = sparse.csr_matrix(rating_matrix)

click_matrix = sparse.lil_matrix((num_users, num_items))
for (u, i) in x_train:
    click_matrix[int(u)-1, int(i)-1] = 1
click_train = sparse.csr_matrix(click_matrix)

x_all = generate_total_sample(num_users, num_items)
y_all = np.array([rating_matrix[u-1,i-1] for (u,i) in x_all])
o_all = np.array([click_matrix[u-1,i-1] for (u,i) in x_all])


#%% TRAIN
model = DRMse(num_users, num_items, embedding_k)
model = model.to(device)

imputator = Imputator(num_users, num_items, embedding_k)
imputator = imputator.to(device)

optimizer_impute = torch.optim.Adam(imputator.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

loss_fcn = lambda x, y, z: F.binary_cross_entropy(x, y, z, reduction="mean")

# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

# one_over_zl, propensity_0, propensity_1 = estimate_ips_bayes(x_train, y_train, y_ips, with_ps=True)
# all_one_over_zl = torch.Tensor([1/propensity_1 if y_ else 1/propensity_0 for y_ in y_all])

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)

    ul_idxs = np.arange(x_all.shape[0])
    np.random.shuffle(ul_idxs)

    model.train()
    imputator.train()

    epoch_total_loss = 0.
    epoch_dr_loss = 0.
    epoch_imputation_loss = 0.

    for idx in range(total_batch):

        '''Imputation Learning'''
        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        # inv_prop = one_over_zl[selected_idx].unsqueeze(-1).to(device)

        imputation, _, __ = imputator(sub_x)
        with torch.no_grad():
            pred_cvr, pred_ctr = model(sub_x)

        inv_prop = 1. / torch.nn.Sigmoid()(pred_ctr)

        true_impute_error = F.binary_cross_entropy(torch.nn.Sigmoid()(pred_cvr), sub_y, reduction="none")

        bias_term = (((imputation - true_impute_error)**2 * tune_lambda * inv_prop) * (1 - torch.nn.Sigmoid()(pred_ctr))**2 / inv_prop**2).mean()
        var_term = (((imputation - true_impute_error)**2 * (1 - tune_lambda) * inv_prop) * (1 - torch.nn.Sigmoid()(pred_ctr)) / inv_prop).mean()

        imputation_loss = bias_term + var_term
        epoch_imputation_loss += imputation_loss

        optimizer_impute.zero_grad()
        imputation_loss.backward()
        optimizer_impute.step()


        '''DR learning'''
        # mini-batch training
        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_all[selected_idx]
        sub_y = y_all[selected_idx]
        sub_o = o_all[selected_idx]

        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
        sub_o = torch.Tensor(sub_o).unsqueeze(-1).to(device)

        # inv_prop = all_one_over_zl[selected_idx].unsqueeze(-1).to(device)

        with torch.no_grad():
            imputation, _, __ = imputator(sub_x)
        pred_cvr, pred_ctr = model(sub_x)

        inv_prop = 1. / torch.nn.Sigmoid()(pred_ctr)

        # true_impute_error = loss_fcn(torch.nn.Sigmoid()(pred), sub_y, None)
        true_impute_error = F.binary_cross_entropy(torch.nn.Sigmoid()(pred_cvr), sub_y, reduction="none")

        ps_loss = loss_fcn(torch.nn.Sigmoid()(pred_ctr), sub_o, None)
        dr_loss = ((true_impute_error - imputation)* sub_o * inv_prop + imputation).mean() + ps_loss
        epoch_dr_loss += dr_loss

        optimizer.zero_grad()
        dr_loss.backward()
        optimizer.step()

    epoch_total_loss = epoch_imputation_loss + epoch_dr_loss
    
    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_imputation_loss': float(epoch_imputation_loss.item()),
        'epoch_dr_loss': float(epoch_dr_loss.item()),
        'epoch_total_loss': float(epoch_total_loss.item()),
    }

    wandb_var.log(loss_dict)

    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = ndcg_func(model, x_test, y_test, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        wandb_var.log(ndcg_dict)

wandb.finish()
