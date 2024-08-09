#%%
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dosnes import dosnes

from module.model import NCF
from module.utils import binarize
from module.similarity import cosine_sim
from module.dataset import load_dataset

#%% SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[3,5,7,10])
parser.add_argument("--balance-param", type=float, default=1.)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset-name", type=str, default="coat")
parser.add_argument("--contrast-pair", type=str, default="both")
parser.add_argument("--base-model", type=str, default="ncf")
parser.add_argument("--sim-measure", type=str, default="cosine")
parser.add_argument("--temperature", type=float, default=2.)

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
balance_param = args.balance_param
data_dir = args.data_dir
dataset_name = args.dataset_name
contrast_pair = args.contrast_pair
base_model = args.base_model
sim_measure = args.sim_measure
temperature = args.temperature


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"

np.random.seed(random_seed)
torch.manual_seed(random_seed)

#%% DATA LOADER
dataset_dir = os.path.join(data_dir, dataset_name)
x_train, x_test = load_dataset(dataset_name, dataset_dir)

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

#%% User/Item similarity
user_user_sim, item_item_sim = cosine_sim(x_train, y_train, num_users, num_items)

user_idx = (user_user_sim != 0).sum(-1).argmax()
alpha = user_user_sim[user_idx,:]

cmap = plt.cm.get_cmap("Spectral")
# colors = np.array([(0.,0.,0.,1.)]+[cmap(i / 4) for i in range(5)])
colors = np.array([cmap(i / 4) for i in range(5)])


user_similar_indices = user_user_sim.topk(5, -1).indices
item_similar_indices = item_item_sim.topk(5, -1).indices

#%%
items_tensor = torch.arange(num_items)
users_tensor = torch.arange(num_users)

ncf_path = '/Users/wonhyung64/Github/DRS/baselines/ncf/ncf.pth'
ncf = NCF(num_users, num_items, embedding_k)
ncf.load_state_dict(torch.load(ncf_path))

ours_path = '/Users/wonhyung64/Github/DRS/ours.pth'
ours = NCF(num_users, num_items, embedding_k)
ours.load_state_dict(torch.load(ours_path), strict=False)

_, ncf_user, __ = ncf(torch.stack([users_tensor, torch.zeros_like(users_tensor)], -1))
_, __, ncf_item = ncf(torch.stack([torch.zeros_like(items_tensor), items_tensor], -1))
ncf_user = ncf_user.detach().numpy()
ncf_item = ncf_item.detach().numpy()

_, ours_user, __ = ours(torch.stack([users_tensor, torch.zeros_like(users_tensor)], -1))
_, __, ours_item = ours(torch.stack([torch.zeros_like(items_tensor), items_tensor], -1))
ours_user = ours_user.detach().numpy()
ours_item = ours_item.detach().numpy()

metric = "sqeuclidean"

model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42, max_iter = 1000)
reduced_ncf_user = model.fit_transform(ncf_user)

model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42, max_iter = 1000)
reduced_ncf_item = model.fit_transform(ncf_item)

model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42, max_iter = 1000)
reduced_ours_user = model.fit_transform(ours_user)

model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42, max_iter = 1000)
reduced_ours_item = model.fit_transform(ours_item)

#%%
u=10
for u in range(num_users):
    sim_pair_u_idx = user_similar_indices[u].numpy()
    colors_u = {k:v for k, v in zip(sim_pair_u_idx, colors)}
    colors_u[u] = np.array([0.,0.,0.,1.])
    all_colors = [colors_u[i] if i in colors_u.keys() else np.array([0.,0.,0.,1.]) for i in range(num_users)]
    all_alpha = [1.0 if i in colors_u.keys() else 0.1 for i in range(num_users)]

    pairs_per_user = np.concatenate([
        np.expand_dims(reduced_ncf_user[u,:], 0),
        reduced_ncf_user[user_similar_indices[u].numpy(),:],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    # ax.scatter(pairs_per_user[:, 0], pairs_per_user[:, 1], pairs_per_user[:, 2], c=colors, cmap=cmap, alpha=1.0)
    ax.scatter(reduced_ncf_user[:, 0], reduced_ncf_user[:, 1], reduced_ncf_user[:, 2], c=all_colors, cmap=cmap, alpha=all_alpha)
    plt.title("Digits Dataset Embedded on a Sphere with metric {}".format(metric))
    plt.colorbar
    plt.show()

    torch.tensor()

    pairs_per_user = np.concatenate([
        np.expand_dims(reduced_ours_user[u,:], 0),
        reduced_ncf_user[user_similar_indices[u].numpy(),:],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(pairs_per_user[:, 0], pairs_per_user[:, 1], pairs_per_user[:, 2], c=colors, cmap=cmap, alpha=1.0)
    plt.title("Digits Dataset Embedded on a Sphere with metric {}".format(metric))
    plt.colorbar
    plt.show()
# %%
import seaborn as sns
sns.color_palette("Spectral", as_cmap=True).

# 5개의 순차적인 색상을 얻기 위해 linspace를 사용하여 색상을 선택

print(colors)
plt.figure(figsize=(8, 2))
for i, color in enumerate(colors):
    plt.bar(i, 1, color=color)

plt.xticks(range(5))
plt.yticks([])
plt.title("Sequential Colors from Spectral Colormap")
plt.show()

#%%
from sklearn.manifold import TSNE
# t-SNE를 사용하여 64차원(이미지의 특성 수)을 2차원으로 축소
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(ncf_user)
X_tsne_ = tsne.fit_transform(ours_user)

# 축소된 데이터를 플롯
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne_[:, 0], X_tsne_[:, 1], c=all_colors, alpha=all_alpha, cmap=cmap)
norm = plt.Normalize(vmin=1, vmax=5)
cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label('digit label')

plt.title('t-SNE Visualization of MNIST Embeddings with Spectral Colorbar')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
plt.colorbar(label='digit label', ticks=range(10))
plt.title('t-SNE Visualization of MNIST Embeddings')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()