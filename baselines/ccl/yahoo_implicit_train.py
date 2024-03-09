#%%
import os
import pdb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset import load_data
from matrix_factorization import NCF, NCF_CVIB, NCF_IPS, NCF_SNIPS, NCF_DR
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU

np.random.seed(2020)
torch.manual_seed(2020)
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)


#%% DATA LOADER
data_dir = "/Users/wonhyung64/Github/DRS/data"
dataset_name = "yahoo_r3"

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


#%% MODEL SETTINGS
embedding_sizes = [4, 8, 16, 32, 64]
hidden_layers_num = [1, 2, 3]
batch_sizes = [512, 1024, 2048, 4096]
balance_params = [0.5, 1.5]
temperatures = [0.1, 1.5]
lrs = [1e-5, 1e-4, 1e-3, 1e-2]
weight_decays = [1e-4, 1e-3, 1e-2]

embedding_k = embedding_sizes[0]
lr = lrs[-1]
weight_decay = weight_decays[-1]
num_epochs = 1000
batch_size = batch_sizes[0]
balance_param = balance_params[0]
temperature = temperatures[0]


#%% MODEL
class NCF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        # self.sigmoid = torch.nn.Sigmoid()
        # self.xent_func = torch.nn.BCELoss()

    def forward(self, x):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)

        h1 = self.linear_1(z_embed)
        h1 = torch.nn.ReLU()(h1)
        out = self.linear_2(h1)

        return out, user_embed, item_embed


model = NCF(num_users, num_items, embedding_k)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()


#%%

num_sample = len(x_train)
total_batch = num_sample // batch_size
model.train()

epoch_loss = 0
for epoch in range(num_epochs):

    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    
    for idx in tqdm(range(total_batch)):
        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1)

        pred, user_embed, item_embed = model(sub_x)

        rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)
        print(rec_loss)

        epoch_loss += rec_loss

        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

    print(epoch_loss / total_batch)


#%%

test_pred = ncf_cvib.predict(x_test)
mse_ncf = mse_func(y_test, test_pred)
auc_ncf = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(ncf_cvib, x_test, y_test)

print("***"*5 + "[NCF-CVIB]" + "***"*5)
print("[NCF-CVIB] test mse:", mse_ncf)
print("[NCF-CVIB] test auc:", auc_ncf)
print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
    np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[NCF-CVIB]" + "***"*5)

# %%
