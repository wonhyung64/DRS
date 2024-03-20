import os
import torch
import numpy as np
import wandb
from datetime import datetime

from model import NCF
from metric import ndcg_func
from utils import binarize, shuffle


for random_seed in range(10):
embedding_sizes = [4, 8, 16, 32, 64]
hidden_layers_num = [1, 2, 3]
batch_sizes = [512, 1024, 2048, 4096]
lrs = [1e-5, 1e-4, 1e-3, 1e-2]
weight_decays = [1e-4, 1e-3, 1e-2]

embedding_k = embedding_sizes[4]
lr = lrs[-2]
weight_decay = weight_decays[0]
batch_size = batch_sizes[2]
num_epochs = 1000
# random_seed = 0
evaluate_interval = 50
top_k_list = [3, 5, 7, 10]

data_dir = "./data"
dataset_name = "yahoo_r3"

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
    }
)
wandb.run.name = f"ncf_{expt_num}"


# DATA LOADER
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


# TRAIN INITIAILIZE
model = NCF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()


# TRAIN
for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    epoch_loss = 0
    model.train()

    for idx in range(total_batch):

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_x = torch.LongTensor(sub_x - 1).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = model(sub_x)

        rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)

        loss_dict: dict = {
            'rec_loss': float(rec_loss.item()),
        }
        wandb_var.log(loss_dict)

        epoch_loss += rec_loss

        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_loss.item():.4f}")

    if epoch % evaluate_interval == 0:
        model.eval()

        ndcg_res = ndcg_func(model, x_test, y_test, device, top_k_list)
        ndcg_dict: dict = {}
        for top_k in top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])
        wandb_var.log(ndcg_dict)
