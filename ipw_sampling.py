#%%
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from module.utils import binarize


#%%
data_dir = "/Users/wonhyung64/Github/DRS/data"
dataset_name = "yahoo_r3"
k = 5

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"


#%% POS NEG MATRIX
data_set_dir = os.path.join(data_dir, dataset_name)
train_file = os.path.join(data_set_dir, "ydata-ymusic-rating-study-v1_0-train.txt")

x_train = []
with open(train_file, "r") as f:
    for line in f:
        x_train.append(line.strip().split())
# <user_id> <song id> <rating>
x_train = np.array(x_train).astype(int)

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()

y_train_ = x_train[:,2:].copy()
bi_x_train = np.concatenate([x_train[:,:2], binarize(y_train_)], axis=-1)


#%% POS MATRIX
total_feedback_list = []
for u in tqdm(range(num_users)):
    user_interactions = bi_x_train[bi_x_train[:,0]-1  == u]
    obs_items = (user_interactions[:,1]-1).tolist()
    obs_feedbacks = (user_interactions[:,2]).tolist()
    user_feedback_list = []
    for item_idx in range(num_items):
        if item_idx in obs_items:
            user_feedback_list.append(obs_feedbacks[obs_items.index(item_idx)])
        else:
            user_feedback_list.append(0)
    total_feedback_list.append(user_feedback_list)
total_pos_feedback = np.array(total_feedback_list).astype(np.float32)

total_pos_interactions = torch.tensor(total_pos_feedback)
total_pos_interactions = total_pos_interactions.to("mps")


#%% NEG MATRIX
total_feedback_list = []
for u in tqdm(range(num_users)):
    user_interactions = bi_x_train[bi_x_train[:,0]-1  == u]
    obs_items = (user_interactions[:,1]-1).tolist()
    obs_feedbacks = (user_interactions[:,2]).tolist()
    user_feedback_list = []
    for item_idx in range(num_items):
        if item_idx in obs_items:
            user_feedback_list.append(obs_feedbacks[obs_items.index(item_idx)] - 1)
        else:
            user_feedback_list.append(0)
    total_feedback_list.append(user_feedback_list)
total_neg_feedback = np.array(total_feedback_list).astype(np.float32)

total_neg_interactions = torch.tensor(total_neg_feedback)
total_neg_interactions = total_neg_interactions.to("mps")


#%% PROPENSITY SCORE
propensity_score = total_pos_interactions.sum(dim=0) / num_users


#%% PREF SIMILARITY
weighted_pos_interactions = total_pos_interactions / propensity_score

pref_user_sim_ = torch.matmul(weighted_pos_interactions, weighted_pos_interactions.T).cpu().numpy()
pref_user_sim = pref_user_sim_ * (np.ones_like(pref_user_sim_) - np.identity(num_users)*2)

pos_user_samples = np.argmax(pref_user_sim, axis=-1) + 1
pref_user_topk = torch.topk(torch.tensor(pref_user_sim), k).indices + 1

pref_item_sim_ = torch.matmul(weighted_pos_interactions.T, weighted_pos_interactions).cpu().numpy()
pref_item_sim = pref_item_sim_ * (np.ones_like(pref_item_sim_) - np.identity(num_items)*2)

pos_item_samples = np.argmax(pref_item_sim, axis=-1) + 1
pref_item_topk = torch.topk(torch.tensor(pref_item_sim), k).indices + 1


#%% PREF DIFFERENCE
weighted_neg_interactions = total_neg_interactions / propensity_score
pref_user_diff_ = torch.matmul(weighted_pos_interactions, weighted_neg_interactions.T).cpu().numpy()
pref_user_diff = pref_user_diff_ * (np.ones_like(pref_user_diff_) - np.identity(num_users)*2)
neg_user_samples = np.argmin(pref_user_diff, axis=-1) + 1

pref_item_diff_ = torch.matmul(weighted_pos_interactions.T, weighted_neg_interactions).cpu().numpy()
pref_item_diff = pref_item_diff_ * (np.ones_like(pref_item_diff_) - np.identity(num_items)*2)
neg_item_samples = np.argmax(pref_item_diff, axis=-1) + 1


#%% SAVE MATRICES
np.save("./assets/ipw_pos_user_samples.npy", pos_user_samples, allow_pickle=True)
np.save("./assets/ipw_pos_user_topk.npy", pref_user_topk.numpy(), allow_pickle=True)

np.save("./assets/ipw_pos_item_samples.npy", pos_item_samples, allow_pickle=True)
np.save("./assets/ipw_pos_item_topk.npy", pref_item_topk.numpy(), allow_pickle=True)

np.save("./assets/ipw_neg_user_samples.npy", neg_user_samples, allow_pickle=True)
np.save("./assets/ipw_neg_item_samples.npy", neg_item_samples, allow_pickle=True)

# %%
