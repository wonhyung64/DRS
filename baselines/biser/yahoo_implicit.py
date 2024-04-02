#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
from datetime import datetime
from scipy import sparse

from model import UAE, IAE
from loss import l2_loss, squred_loss
from metric import biser_ndcg_func
from utils import binarize

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--uae-latent-dim", type=int, default=50)
parser.add_argument("--iae-latent-dim", type=int, default=50)
parser.add_argument("--uae-lr", type=float, default=0.1)
parser.add_argument("--iae-lr", type=float, default=0.2)
parser.add_argument("--uae-batch-size", type=int, default=4)
parser.add_argument("--iae-batch-size", type=int, default=1)
parser.add_argument("--uae-l2-lambda", type=float, default=0.)
parser.add_argument("--iae-l2-lambda", type=float, default=0.)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../../data")
parser.add_argument("--dataset-name", type=str, default="yahoo_r3")


try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


uae_latent_dim = args.uae_latent_dim
iae_latent_dim = args.iae_latent_dim
uae_lr = args.uae_lr
iae_lr = args.iae_lr
uae_batch_size = args.uae_batch_size
iae_batch_size = args.iae_batch_size
num_epochs = args.num_epochs
uae_l2_lambda =  args.uae_l2_lambda
iae_l2_lambda =  args.iae_l2_lambda
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

np.random.seed(random_seed)
torch.manual_seed(random_seed)


#%% WANDB
wandb_var = wandb.init(
    project="drs",
    config={
        "device" : device,
        "uae_latent_dim" : uae_latent_dim,
        "iae_latent_dim" : iae_latent_dim,
        "uae_batch_size" : uae_batch_size,
        "iae_batch_size" : iae_batch_size,
        "num_epochs" : num_epochs,
        "evaluate_interval" : evaluate_interval,
        "uae_l2_lambda" : uae_l2_lambda,
        "iae_l2_lambda" : iae_l2_lambda,
        "uae_lr" : uae_lr,
        "iae_lr" : iae_lr,
        "top_k_list" : top_k_list,
        "random_seed" : random_seed,
    }
)
wandb.run.name = f"biser_{expt_num}"


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

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print("# user: {}, # item: {}".format(num_users, num_items))

x_train, y_train = x_train[:,:-1], x_train[:,-1:]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

y_train = binarize(y_train)
y_test = binarize(y_test)


matrix = sparse.lil_matrix((num_users, num_items))
for (u, i, r) in np.concatenate([x_train, y_train], axis=-1):
    if r:
        matrix[int(u)-1, int(i)-1] = r
sparse_train = sparse.csr_matrix(matrix)

train_ui_matrix = sparse_train.toarray()
train_iu_matrix = sparse_train.T.toarray()


#%%
train_dict = {}
for idx, value in enumerate(sparse_train):
    train_dict[idx] = value.indices.copy().tolist()

total_batch = num_users // batch_size

#%% TRAIN
uae = UAE(num_users, num_items)
uae = uae.to(device)

iae = IAE(num_users, num_items)
iae = iae.to(device)

uae_optimizer = torch.optim.Adagrad(uae.parameters(), lr=uae_lr)
iae_optimizer = torch.optim.Adagrad(iae.parameters(), lr=iae_lr)


for epoch in range(1, num_epochs+1):
    uae.train()
    if type(train_ui_matrix) != torch.Tensor:
        train_ui_matrix = torch.LongTensor(train_ui_matrix).type(torch.float32).to(device)
    pred_uae_matrix, _ = uae(train_ui_matrix)

    iae.train()
    if type(train_iu_matrix) != torch.Tensor:
        train_iu_matrix = torch.LongTensor(train_iu_matrix).type(torch.float32).to(device)
    pred_iae_matrix, _ = iae(train_iu_matrix)

    all_user_idx = np.arange(num_users)
    all_item_idx = np.arange(num_items)
    np.random.shuffle(all_user_idx)
    np.random.shuffle(all_item_idx)

    epoch_loss = 0
    #UAE
    total_batch = num_users // uae_batch_size
    for idx in range(total_batch):
        # mini-batch training
        selected_idx = all_user_idx[uae_batch_size*idx : (idx+1)*uae_batch_size]

        batch_pred = np.zeros((uae_batch_size, num_items))
        batch_true = np.zeros((uae_batch_size, num_items))
        for i, user_id in enumerate(selected_idx):
            batch_pred[i] = pred_uae_matrix[user_id]
            sub_x[i, items_by_user_id] = 1

        sub_x = torch.LongTensor(sub_x).type(torch.float32).to(device)
        pred, z = model(sub_x)

        recon_loss = squred_loss(sub_x, pred)
        l2_reg = l2_loss(model) * l2_lambda
        total_loss = recon_loss + l2_reg

        loss_dict: dict = {
            'recon_loss': float(recon_loss.item()),
            'l2_reg': float(l2_reg.item()),
            'total_loss': float(total_loss.item()),
        }
        wandb_var.log(loss_dict)

        epoch_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_loss.item():.4f}")

    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = uae_ndcg_func(model, x_test, y_test, train_dict, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        wandb_var.log(ndcg_dict)

wandb.finish()
