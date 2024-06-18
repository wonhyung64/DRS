import os
import torch
import numpy as np
from tqdm import tqdm
from .utils import binarize


def seperate_pos_neg_interactions(x_train, dataset_name):
    num_users = x_train[:,0].max()
    num_items = x_train[:,1].max()
    y_train_ = x_train[:,2:].copy()
    bi_x_train = np.concatenate([x_train[:,:2], binarize(y_train_)], axis=-1)

    """POS MATRIX"""
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

    """NEG MATRIX"""
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

    os.makedirs(f"./assets/{dataset_name}", exist_ok=True)
    np.save(f"./assets/{dataset_name}/pos_interactions.npy", total_pos_feedback, allow_pickle=True)
    np.save(f"./assets/{dataset_name}/neg_interactions.npy", total_neg_feedback, allow_pickle=True)


def compute_sim_matrix(pos_interactions, num_users, num_items, k=5):
    pref_user_sim_ = torch.matmul(pos_interactions, pos_interactions.T).cpu().numpy()
    pref_user_sim = pref_user_sim_ * (np.ones_like(pref_user_sim_) - np.identity(num_users)*2)
    pref_user_topk = torch.topk(torch.tensor(pref_user_sim), k).indices + 1

    pref_item_sim_ = torch.matmul(pos_interactions.T, pos_interactions).cpu().numpy()
    pref_item_sim = pref_item_sim_ * (np.ones_like(pref_item_sim_) - np.identity(num_items)*2)
    pref_item_topk = torch.topk(torch.tensor(pref_item_sim), k).indices + 1

    return pref_user_topk, pref_item_topk
