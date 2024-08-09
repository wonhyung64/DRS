#%%
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from tqdm import tqdm

from baselines.ncf.model import NCF
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

cmap = plt.cm.viridis
# colors = np.array([(0.,0.,0.,1.)]+[cmap(i / 4) for i in range(5)])
colors = np.array([cmap(i / 4) for i in range(5)])


user_similar_indices = user_user_sim.topk(5, -1).indices
item_similar_indices = item_item_sim.topk(5, -1).indices

#%% EMBEDDING VECTORS
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


#%%
tsne = TSNE(n_components=2, random_state=42)
reduced_ncf_user = tsne.fit_transform(ncf_user)

tsne = TSNE(n_components=2, random_state=42)
reduced_ncf_item = tsne.fit_transform(ncf_item)

tsne = TSNE(n_components=2, random_state=42)
reduced_ours_user = tsne.fit_transform(ours_user)

tsne = TSNE(n_components=2, random_state=42)
reduced_ours_item = tsne.fit_transform(ours_item)


#%%
for u in tqdm(range(num_users)):
    sim_pair_u_idx = user_similar_indices[u].numpy()
    left_users = np.random.choice([user for user in range(num_users) if (user not in sim_pair_u_idx) and user!=u], 24)
    sub_users = np.concatenate([
        sim_pair_u_idx,
        left_users,
    ])
    sub_colors = np.concatenate([
        colors,
        np.array([0.,0.,0.,1.]*24).reshape(24,4),
    ])
    sub_alphas =np.array([1.]*5 + [0.1]*24)
    sub_markers = ["o"]*5 + ["o"]*24

    plt.rcParams.update({
        'font.family': 'serif',         # Serif 폰트 계열 사용
        'font.serif': ['Times New Roman'],  # Times New Roman 폰트 사용
        'text.usetex': False,           # LaTeX 사용 안 함
        'axes.labelsize': 12,           # 축 레이블 글자 크기
        'axes.titlesize': 14,           # 축 제목 글자 크기
        'xtick.labelsize': 10,          # x축 눈금 글자 크기
        'ytick.labelsize': 10,          # y축 눈금 글자 크기
        'legend.fontsize': 10,          # 범례 글자 크기
        'figure.titlesize': 16          # 전체 제목 글자 크기
    })
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1], width_ratios=[1, 1])

    # Subplot 생성
    ax1 = plt.subplot(gs[0, 0])  # 왼쪽 subplot
    ax2 = plt.subplot(gs[0, 1])  # 오른쪽 subplot
    cbar_ax = plt.subplot(gs[1, :])  # 전체 넓이를 사용하는 colorbar 영역

    # Figure 배경색 설정
    fig.patch.set_facecolor('#f0f0f0')

    # Axes 배경색 설정
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    # 두 subplot에 데이터 플롯

    ax1.scatter(reduced_ncf_user[sub_users, 0], reduced_ncf_user[sub_users, 1], color=sub_colors, alpha=sub_alphas, marker="o", s=200)
    ax1.scatter(reduced_ncf_user[u, 0], reduced_ncf_user[u, 1], color=[0.,0.,0.,1.], alpha=1., marker="x", s=400)
    # ax1.scatter(reduced_ncf_user[u, 0], reduced_ncf_user[u, 1], c="red", alpha=1., marker="x", s=300)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("NCF", fontsize=20)

    ax2.scatter(reduced_ours_user[sub_users, 0], reduced_ours_user[sub_users, 1], color=sub_colors, alpha=sub_alphas, marker="o", s=200)
    ax2.scatter(reduced_ours_user[u, 0], reduced_ours_user[u, 1], color=[0.,0.,0.,1.], alpha=1., marker="x", s=400)
    # ax2.scatter(reduced_ours_user[u, 0], reduced_ours_user[u, 1], c="red", alpha=1., marker="x", s=300)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Ours", fontsize=20)

    # Colorbar를 'Spectral' colormap으로 생성, 위치를 아래로 설정
    cmap_spectral = plt.cm.viridis
    norm = plt.Normalize(vmin=1, vmax=5)
    cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap_spectral), cax=cbar_ax, orientation='horizontal')

    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels([1, 2, 3, 4, 5], fontsize=18)
    cbar.set_label('Similarity Rank', fontsize=18)

    fig.tight_layout(rect=[0, 0.15, 1, 1])  # rect 파라미터를 사용하여 전체 figure의 여백 조정
    fig.savefig(f"/Users/wonhyung64/Github/DRS/results/tsne/user/user_{'{:03d}'.format(u)}")
    plt.close()
# %%
for i in tqdm(range(num_items)):
    sim_pair_i_idx = item_similar_indices[i].numpy()
    left_items = np.random.choice([item for item in range(num_items) if (item not in sim_pair_i_idx) and item!=i], 24)
    sub_items = np.concatenate([
        sim_pair_i_idx,
        left_items,
    ])
    sub_colors = np.concatenate([
        colors,
        np.array([0.,0.,0.,1.]*24).reshape(24,4),
    ])
    sub_alphas =np.array([1.]*5 + [0.1]*24)
    sub_markers = ["o"]*5 + ["o"]*24

    plt.rcParams.update({
        'font.family': 'serif',         # Serif 폰트 계열 사용
        'font.serif': ['Times New Roman'],  # Times New Roman 폰트 사용
        'text.usetex': False,           # LaTeX 사용 안 함
        'axes.labelsize': 12,           # 축 레이블 글자 크기
        'axes.titlesize': 14,           # 축 제목 글자 크기
        'xtick.labelsize': 10,          # x축 눈금 글자 크기
        'ytick.labelsize': 10,          # y축 눈금 글자 크기
        'legend.fontsize': 10,          # 범례 글자 크기
        'figure.titlesize': 16          # 전체 제목 글자 크기
    })
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1], width_ratios=[1, 1])

    # Subplot 생성
    ax1 = plt.subplot(gs[0, 0])  # 왼쪽 subplot
    ax2 = plt.subplot(gs[0, 1])  # 오른쪽 subplot
    cbar_ax = plt.subplot(gs[1, :])  # 전체 넓이를 사용하는 colorbar 영역

    # Figure 배경색 설정
    fig.patch.set_facecolor('#f0f0f0')

    # Axes 배경색 설정
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    # 두 subplot에 데이터 플롯

    ax1.scatter(reduced_ncf_item[sub_items, 0], reduced_ncf_item[sub_items, 1], color=sub_colors, alpha=sub_alphas, marker="o", s=200)
    ax1.scatter(reduced_ncf_item[i, 0], reduced_ncf_item[i, 1], color=[0.,0.,0.,1.], alpha=1., marker="x", s=400)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("NCF", fontsize=20)

    ax2.scatter(reduced_ours_item[sub_items, 0], reduced_ours_item[sub_items, 1], color=sub_colors, alpha=sub_alphas, marker="o", s=200)
    ax2.scatter(reduced_ours_item[i, 0], reduced_ours_item[i, 1], color=[0.,0.,0.,1.], alpha=1., marker="x", s=400)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Ours", fontsize=20)

    # Colorbar를 'Spectral' colormap으로 생성, 위치를 아래로 설정
    cmap_spectral = plt.cm.viridis
    norm = plt.Normalize(vmin=1, vmax=5)
    cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap_spectral), cax=cbar_ax, orientation='horizontal')

    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels([1, 2, 3, 4, 5], fontsize=18)
    cbar.set_label('Similarity Rank', fontsize=18)

    fig.tight_layout(rect=[0, 0.15, 1, 1])  # rect 파라미터를 사용하여 전체 figure의 여백 조정
    fig.savefig(f"/Users/wonhyung64/Github/DRS/results/tsne/item/item_{'{:03d}'.format(i)}")
    plt.close()

# %%
