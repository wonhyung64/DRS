import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

from model import NCF
from metric import ndcg_func
from utils import binarize, shuffle

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=2048)
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
    }
)
wandb.run.name = f"ncf_{expt_num}"


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
    x_test = pd.read_csv(test_file).to_numpy()


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


#%% TRAIN
model = NCF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    epoch_total_loss = 0.
    epoch_rec_loss = 0.

    for idx in range(total_batch):

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = model(sub_x)

        rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)
        epoch_rec_loss += rec_loss

        total_loss = rec_loss
        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_rec_loss': float(epoch_rec_loss.item()),
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
