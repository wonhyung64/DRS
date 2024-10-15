import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict


def ndcg_func(model, x_test, y_test, device, top_k_list):
    """
    Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        pred_ = model(x_u)
        pred = pred_.flatten().cpu().detach()

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            true_top_k_rel = y_u[np.argsort(-y_u)][:top_k]
            dcg_k = (2**pred_top_k_rel-1) / log2_iplus1
            idcg_k = (2**true_top_k_rel-1) / log2_iplus1

            if np.sum(idcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(idcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map


def recall_func(model, x_test, y_test, device, top_k_list):
    """
    Evaluate recall@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        pred_ = model(x_u)
        pred = pred_.flatten().cpu().detach()
        total_rel = sum(y_u == 1)

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            recall_k = sum(pred_top_k_rel) / total_rel

            if total_rel == 0:
                recall_k = 1.

            result_map[f"recall_{top_k}"].append(recall_k)

    return result_map


def ap_func(model, x_test, y_test, device, top_k_list):
    """
    Evaluate ap@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        pred_ = model(x_u)
        pred = pred_.flatten().cpu().detach()

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            N = sum(pred_top_k_rel)
            precision_k = np.cumsum(pred_top_k_rel) / np.arange(1, top_k+1)
            ap_k = np.sum(precision_k * pred_top_k_rel) / N

            if N == 0:
                ap_k = 1.

            result_map[f"ap_{top_k}"].append(ap_k)

    return result_map


def ndcg_func_ssm(model, x_test, y_test, device, top_k_list):
    """
    Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        _, user_embed, item_embed = model(x_u)
        user_norm = F.normalize(user_embed, p=2, dim=1)
        item_norm = F.normalize(item_embed, p=2, dim=1)
        pred_ = (user_norm * item_norm).sum(-1)
        pred = pred_.flatten().cpu().detach()

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            true_top_k_rel = y_u[np.argsort(-y_u)][:top_k]
            dcg_k = (2**pred_top_k_rel-1) / log2_iplus1
            idcg_k = (2**true_top_k_rel-1) / log2_iplus1

            if np.sum(idcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(idcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map


def recall_func_ssm(model, x_test, y_test, device, top_k_list):
    """
    Evaluate recall@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        _, user_embed, item_embed = model(x_u)
        user_norm = F.normalize(user_embed, p=2, dim=1)
        item_norm = F.normalize(item_embed, p=2, dim=1)
        pred_ = (user_norm * item_norm).sum(-1)
        pred = pred_.flatten().cpu().detach()
        total_rel = sum(y_u == 1)

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            recall_k = sum(pred_top_k_rel) / total_rel

            if total_rel == 0:
                recall_k = 1.

            result_map[f"recall_{top_k}"].append(recall_k)

    return result_map


def ap_func_ssm(model, x_test, y_test, device, top_k_list):
    """
    Evaluate ap@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_item_idx = all_tr_idx[x_test[:, 0] == uid]
        x_u = torch.LongTensor(x_test[u_item_idx]-1).to(device)
        y_u = y_test[u_item_idx]
        _, user_embed, item_embed = model(x_u)
        user_norm = F.normalize(user_embed, p=2, dim=1)
        item_norm = F.normalize(item_embed, p=2, dim=1)
        pred_ = (user_norm * item_norm).sum(-1)
        pred = pred_.flatten().cpu().detach()

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            pred_top_k_rel = y_u[np.argsort(-pred.numpy())][:top_k]
            N = sum(pred_top_k_rel)
            precision_k = np.cumsum(pred_top_k_rel) / np.arange(1, top_k+1)
            ap_k = np.sum(precision_k * pred_top_k_rel) / N

            if N == 0:
                ap_k = 1.

            result_map[f"ap_{top_k}"].append(ap_k)

    return result_map
