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


def corr_sim(x_train, y_train, num_users: int, num_items: int):
    total_feedback_list = []
    for u in tqdm(range(1, num_users+1)):
        u_idxs = x_train[:,0] == u
        obs_items = x_train[u_idxs, 1]
        obs_feedbacks = y_train[u_idxs]

        user_feedback_list = []
        for i in range(1, num_items+1):
            if i in obs_items:
                user_feedback_list.append(obs_feedbacks[obs_items==i][0])
            else:
                user_feedback_list.append(0)
        total_feedback_list.append(user_feedback_list)
    total_pos_feedback = np.array(total_feedback_list).astype(np.float32) + 0.1

    user_user_sim = torch.tensor(total_pos_feedback).corrcoef() * (1 - torch.eye(num_users))
    item_item_sim = torch.tensor(total_pos_feedback).T.corrcoef() * (1 - torch.eye(num_items))

    return user_user_sim, item_item_sim


def cosine_sim(x_train, y_train, num_users: int, num_items: int):
    total_feedback_list = []
    for u in tqdm(range(1, num_users+1)):
        u_idxs = x_train[:,0] == u
        obs_items = x_train[u_idxs, 1]
        obs_feedbacks = y_train[u_idxs]

        user_feedback_list = []
        for i in range(1, num_items+1):
            if i in obs_items:
                user_feedback_list.append(obs_feedbacks[obs_items==i][0])
            else:
                user_feedback_list.append(0)
        total_feedback_list.append(user_feedback_list)
    total_pos_feedback = torch.tensor(np.array(total_feedback_list).astype(np.float32))
    
    user_norm = total_pos_feedback.T.norm(dim=0, p=2).maximum(torch.tensor(1e-8))
    item_norm = total_pos_feedback.norm(dim=0, p=2).maximum(torch.tensor(1e-8))

    user_user_sim = torch.matmul(total_pos_feedback, total_pos_feedback.T) / user_norm.outer(user_norm) * (1 - torch.eye(num_users))
    item_item_sim = torch.matmul(total_pos_feedback.T, total_pos_feedback) / item_norm.outer(item_norm) * (1 - torch.eye(num_items))

    return user_user_sim, item_item_sim
