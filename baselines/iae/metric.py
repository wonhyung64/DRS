import torch
import numpy as np
from collections import defaultdict


def iae_ndcg_func(model, x_test, y_test, train_dict, device, top_k_list):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_item_ids = np.unique(x_test[:,1])
    all_tr_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for iid in all_item_ids:
        item_idx = all_tr_idx[x_test[:,1] == iid]
        user_idx = x_test[item_idx, 0] - 1
        y_i = y_test[item_idx]

        sub_x = np.zeros((1, model.num_users))
        users_by_iid = train_dict[iid-1]
        sub_x[0, users_by_iid] = 1

        sub_x = torch.LongTensor(sub_x).type(torch.float32).to(device)
        pred_, _ = model(sub_x)
        pred_ = pred_.flatten().cpu().detach()
        pred = pred_[user_idx]

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred.numpy())[:top_k]
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            dcg_k = y_i[pred_top_k] / log2_iplus1
            best_dcg_k = y_i[np.argsort(-y_i)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map
