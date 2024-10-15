import torch
import numpy as np
from collections import defaultdict


def biser_ndcg_func(uae, iae, x_test, y_test, train_ui_matrix, train_iu_matrix, device, top_k_list):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_ids = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    pred_uae_matrix, _ = uae(train_ui_matrix)
    pred_iae_matrix, _ = iae(train_iu_matrix)
    pred_matrix = (pred_uae_matrix + pred_iae_matrix.T)/2

    for uid in all_user_ids:
        user_idx = all_tr_idx[x_test[:,0] == uid]
        item_idx = x_test[user_idx, 1] - 1
        y_u = y_test[user_idx]

        pred = pred_matrix.cpu().detach()[uid-1, item_idx]

        for top_k in top_k_list:
            if len(y_u) < top_k:
                break
            pred_top_k = np.argsort(-pred.numpy())[:top_k]
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            dcg_k = y_u[pred_top_k] / log2_iplus1
            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map
