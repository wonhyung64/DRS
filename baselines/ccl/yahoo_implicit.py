#%%
import os
import wandb
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from model import NCF
from metric import ndcg_func
from utils import binarize, shuffle


#%% SETTINGS
embedding_sizes = [4, 8, 16, 32, 64]
hidden_layers_num = [1, 2, 3]
batch_sizes = [512, 1024, 2048, 4096]
balance_params = [0.5, 1.5]
temperatures = [0.1, 1.5]
lrs = [1e-5, 1e-4, 1e-3, 1e-2]
weight_decays = [1e-4, 1e-3, 1e-2]
sampling = "cf"

embedding_k = embedding_sizes[4]
lr = lrs[-2]
weight_decay = weight_decays[0]
batch_size = batch_sizes[2]
num_epochs = 1000
random_seed = 0
evaluate_interval = 50
top_k_list = [3, 5, 7]
balance_param = balance_params[0]
temperature = temperatures[0]

data_dir = "/Users/wonhyung64/Github/DRS/data"
dataset_name = "yahoo_r3"

if torch.backends.mps.is_available():
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
        "weight_decat": weight_decay,
        "top_k_list" : top_k_list,
        "random_seed" : random_seed,
        "sampling" : sampling,
        "balance_param" : balance_param,
        "temperature" : temperature,
    }
)
wandb.run.name = f"ccl_{expt_num}"


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


x_train, y_train = shuffle(x_train, y_train)
num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print("# user: {}, # item: {}".format(num_users, num_items))

y_train = binarize(y_train)
y_test = binarize(y_test)

num_sample = len(x_train)
total_batch = num_sample // batch_size

train_df = pd.DataFrame(data=x_train, columns=["user", "item"])
train_popularity = train_df["item"].value_counts().reset_index().sort_values("item")["count"].values
train_popularity = np.sqrt(train_popularity / train_popularity.max())
pop_difference = np.abs(train_popularity.repeat((num_items)).reshape(num_items, -1) - train_popularity)
top_pop_diffs = np.argmax(pop_difference, -1)+1

unexposed_dict = {}
for i in range(1, num_users+1):
    exposed_items = train_df[train_df["user"] == i]["item"].tolist()
    unexposed_items = [j for j in range(1, num_items+1) if j not in exposed_items]
    unexposed_dict[i] = unexposed_items


#%% TRAIN INITIAILIZE
model = NCF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()


#%% TRAIN
for epoch in range(1, num_epochs+1):break
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    epoch_loss = 0
    model.train()

    for idx in range(total_batch):break
        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        # sampling
        batch_users = x_train[selected_idx, 0].tolist()

        if sampling == "cf":
            unexposed_items_ = []
            for u in batch_users:
                unexposed_item = np.random.choice(unexposed_dict[u],1)
                unexposed_items_.append(unexposed_item)
            augmented_items = torch.tensor(np.stack(unexposed_items_)-1).to(device)

        elif sampling == "pop":
            top_pop_diffs

        aug_x = torch.cat([sub_x[:, :1], augmented_items], dim=-1)

        pred, user_embed, item_embed = model(sub_x)
        _, aug_user_embed, aug_item_embed = model(aug_x)
        org_embed = torch.cat([user_embed, item_embed], dim=-1)
        aug_embed = torch.cat([user_embed, item_embed], dim=-1)

        rev_identity = torch.LongTensor(1 - np.identity(batch_size))
        indicator = torch.cat([rev_identity, torch.ones_like(rev_identity)], dim=-1).to(device)

        ccl_loss_ = (org_embed * aug_embed).sum(-1)
        total_embed = torch.cat([org_embed, aug_embed], dim=0)
        norm_ccl_ = (torch.matmul(org_embed, total_embed.T) * temperature).exp()
        norm_ccl = (norm_ccl_ * indicator).sum(-1)
        ccl_loss = -torch.log(ccl_loss_ / norm_ccl).mean()
        
        rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)

        total_loss = rec_loss + ccl_loss

        loss_dict: dict = {
            'rec_loss': float(rec_loss.item()),
            'ccl_loss': float(ccl_loss.item()),
            'total_loss': float(total_loss.item()),
        }
        wandb_var.log(loss_dict)

        epoch_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_loss.item():.4f}")

    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = ndcg_func(model, x_test, y_test, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        wandb_var.log(ndcg_dict)

# %%
