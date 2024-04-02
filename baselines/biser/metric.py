import torch
import numpy as np
from collections import defaultdict


def biser_ndcg_func(model, x_test, y_test, train_dict, device, top_k_list):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_ids = np.unique(x_test[:,0])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for uid in all_user_ids:
        user_idx = all_tr_idx[x_test[:,0] == uid]
        item_idx = x_test[user_idx, 1] - 1
        y_u = y_test[user_idx]

        sub_x = np.zeros((1, model.num_items))
        items_by_uid = train_dict[uid-1]
        sub_x[0, items_by_uid] = 1

        sub_x = torch.LongTensor(sub_x).type(torch.float32).to(device)
        pred_, _ = model(sub_x)
        pred_ = pred_.flatten().cpu().detach()
        pred = pred_[item_idx]

        for top_k in top_k_list:
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
