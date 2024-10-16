#%%
WANDB_TRACKING = 0
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

from module.model import MF, NCF
from module.utils import binarize

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb
finally: 
    WANDB_TRACKING = 1


#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--preference-factor-dim", type=int, default=4)
parser.add_argument("--preference-lr", type=float, default=1e-2)
parser.add_argument("--preference-weight-decay", type=float, default=1e-4)
parser.add_argument("--preference-batch-size", type=int, default=2048)
parser.add_argument("--preference-num-epochs", type=int, default=1000)
parser.add_argument("--preference-base-model", type=str, default="MF")

parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset-name", type=str, default="ml-1m") # [coat, kuairec, yahoo_r3, ml-1m]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

preference_factor_dim = args.preference_factor_dim
preference_lr = args.preference_lr
preference_weight_decay = args.preference_weight_decay
preference_batch_size = args.preference_batch_size
preference_num_epochs = args.preference_num_epochs
preference_base_model = args.preference_base_model

random_seed = args.random_seed
data_dir = args.data_dir
dataset_name = args.dataset_name

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
save_dir = f"./weights/preference/expt_{expt_num}"
os.makedirs(f"{save_dir}", exist_ok=True)

config = vars(args)
config["device"] = device
config["expt_num"] = expt_num
config["save_dir"] = save_dir

if WANDB_TRACKING:
    wandb_var = wandb.init(project="recommender", config=config)
    wandb.run.name = f"pref_{expt_num}"


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
print("# prefer: {}, # not prefer: {}".format(y_train.sum(), num_sample - y_train.sum()))


#%% LOAD MODEL
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if preference_base_model == "MF":
    preference_model = MF(num_users, num_items, preference_factor_dim)
elif preference_base_model == "NCF":
    preference_model = NCF(num_users, num_items, preference_factor_dim)
preference_model = preference_model.to(device)
optimizer = torch.optim.Adam(preference_model.parameters(), lr=preference_lr, weight_decay=preference_weight_decay)
loss_fcn = torch.nn.BCELoss()

#%%
np.random.seed(random_seed)
torch.manual_seed(random_seed)

all_idx = np.arange(num_sample)
for epoch in range(1, preference_num_epochs+1):
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

    if WANDB_TRACKING:
        loss_dict: dict = {
            'epoch_preference_loss': float(epoch_preference_loss.item()),
        }
        wandb_var.log(loss_dict)

torch.save(preference_model, f"{save_dir}/e{epoch}.pth")
