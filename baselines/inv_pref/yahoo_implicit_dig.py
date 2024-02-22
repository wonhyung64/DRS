#%%
import os
import math
import json
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

from model import InvPrefImplicit, _init_eps

# from dataloader import BaseImplicitBCELossDataLoader, YahooImplicitBCELossDataLoader

# from evaluate import ImplicitTestManager
# from models import InvPrefImplicit
# from train import ImplicitTrainManager
# from utils import draw_score_pic, merge_dict, _show_me_a_list_func, draw_loss_pic_one_by_one, query_user, query_str, \
#     mkdir, query_int, get_class_name_str, _mean_merge_dict_func, show_me_all_the_fucking_result


def analyse_interaction_from_text(lines: list, has_value: bool = False):

    pairs: list = []

    users_set: set = set()
    items_set: set = set()

    for line in tqdm(lines):
        elements: list = line.split(',')
        user_id: int = int(elements[0])
        item_id: int = int(elements[1])
        if not has_value:
            pairs.append([user_id, item_id])
        else:
            value: float = float(elements[2])
            pairs.append([user_id, item_id, value])

        users_set.add(user_id)
        items_set.add(item_id)

    users_list: list = list(users_set)
    items_list: list = list(items_set)

    users_list.sort(reverse=False)
    items_list.sort(reverse=False)

    return pairs, users_list, items_list


def analyse_user_interacted_set(pairs: list):
    user_id_list: list = list()
    print('Init table...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        # user_bought_map.append(set())
        user_id_list.append(user_id)

    max_user_id: int = max(user_id_list)
    user_bought_map: list = [set() for i in range((max_user_id + 1))]
    print('Build mapping...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        user_bought_map[user_id].add(item_id)

    return user_bought_map


def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def evaluate(
    model, test_users_tensor, test_user_list, sorted_ground_truth,
    batch_size, user_positive_interaction, top_k_list, use_item_pool, item_pool) -> dict:
    model.eval()

    def _merge_dicts_elements_func(elements_list, **args):
        user_num: int = args['user_num']
        return (np.sum(np.array(elements_list), axis=0) / float(user_num)).tolist()

    all_test_user_tensor: torch.Tensor = test_users_tensor
    all_test_user_list: list = test_user_list
    all_test_ground_truth: list = sorted_ground_truth

    result_dicts_list: list = []

    for (batch_index, (batch_users_tensor, batch_users_list, batch_users_ground_truth)) \
            in enumerate(mini_batch(batch_size, all_test_user_tensor, all_test_user_list,
                                    all_test_ground_truth)):

        rating_matrix: torch.Tensor = model.predict(batch_users_tensor)

        mask_users, mask_items = [], []

        for idx, user_id in enumerate(batch_users_list):
            mask_items_set = user_positive_interaction[user_id]
            mask_users += [idx] * len(mask_items_set)
            mask_items += list(mask_items_set)
        rating_matrix[mask_users, mask_items] = -(1 << 10)

        if use_item_pool:
            high_light_users, high_light_items = [], []
            for idx, user_id in enumerate(batch_users_list):
                high_light_items_set = item_pool[user_id]
                high_light_users += [idx] * len(high_light_items_set)
                high_light_items += list(high_light_items_set)
            rating_matrix[high_light_users, high_light_items] += (1 << 10)
            

        # print(rating_matrix.shape)
        _, predict_items = torch.topk(rating_matrix, k=max(top_k_list))
        predict_items: np.array = predict_items.cpu().numpy()
        # print(predict_items)
        # print(type(predict_items))
        # print(type(predict_items[0]))
        predict_items_list: list = predict_items.tolist()

        r = get_label(batch_users_ground_truth, predict_items_list)

        pre, recall, ndcg = [], [], []
        for k in top_k_list:
            ret = recall_precision_ATk(batch_users_ground_truth, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ret = NDCGatK_r(batch_users_ground_truth, r, k)
            ndcg.append(ret)

        result_dict: dict = {
            'ndcg': ndcg,
            'recall': recall,
            'precision': pre,
        }

        result_dicts_list.append(result_dict)

    # print(result_dicts_list)
    result_dict: dict = merge_dict(
        dict_list=result_dicts_list,
        merge_func=_merge_dicts_elements_func,
        user_num=len(all_test_user_list)
    )

    reformat_result_dict: dict = {}
    for metric in result_dict.keys():
        values: list = result_dict[metric]
        metric_result_dict: dict = {}
        for idx, value in enumerate(values):
            metric_result_dict[top_k_list[idx]] = values[idx]
        reformat_result_dict[metric] = metric_result_dict

    return reformat_result_dict


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recall_precision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = np.array([k for i in range(len(test_data))])
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred / precis_n)

    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)



def merge_dict(dict_list: list, user_num: int):
    # assert len(dict_list) > 1, 'len(dict_list) should bigger than 1'
    first_dict: dict = dict_list[0]
    keys = first_dict.keys()
    for element_dict in dict_list:
        assert keys == element_dict.keys()

    result: dict = dict()
    for key in keys:
        elements_list: list = [element_dict[key] for element_dict in dict_list]
        result[key] = (np.sum(np.array(elements_list), axis=0) / float(user_num)).tolist()

    return result


#%%
expt_dir = "/Users/wonhyung64/Github/DRS/baselines/inv_pref/weights/expt_240222_210936_733741"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = "mps"
# MODEL_CONFIG
env_num = 2
factor_num = 40
reg_only_embed = True
reg_env_embed = False

# EVALUATE_CONFIG
top_k_list = [3, 5, 7]
test_batch_size = 1024
eval_k = 5
eval_metric = "ndcg"

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]
DATASET_PATH = '/Yahoo_all_data/'
METRIC_LIST = ['ndcg', 'recall', 'precision']

# expt config
random_seed = 0
has_item_pool: bool = True
cluster_use_random_sort: bool = False # org True


#%% DataLoader
dataset_path = "/Users/wonhyung64/Github/DRS/data/yahoo_r3/implicit"

train_data_path: str = f"{dataset_path}/train.csv"
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]
_train_data: np.array = train_df.values.astype(np.int64)

test_data_path: str = f"{dataset_path}/test.csv"
test_df: pd.DataFrame = pd.read_csv(test_data_path)
_test_data: np.array = test_df.values.astype(np.int64)


user_positive_interaction = []
user_list: list = []
item_list: list = []

test_user_list: list = []
test_item_list: list = []
ground_truth: list = []


with open(train_data_path, 'r') as inp:
    inp.readline()
    lines: list = inp.readlines()

    print('Begin analyze raw train file')
    pairs, user_list, item_list = analyse_interaction_from_text(lines, has_value=True)

    positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))

    user_positive_interaction: list = analyse_user_interacted_set(positive_pairs) # observed intercation cases for all 15400 users

    _train_pairs: list = pairs

    inp.close()

with open(test_data_path, 'r') as inp:
    inp.readline()
    lines: list = inp.readlines()
    print('Begin analyze raw test file')
    pairs, test_user_list, test_item_list = analyse_interaction_from_text(lines)
    # print(self.test_user_list)
    ground_truth: list = analyse_user_interacted_set(pairs)
    inp.close()

if has_item_pool:
    item_pool_path: str = dataset_path + '/test_item_pool.csv'
    with open(item_pool_path, 'r') as inp:
        inp.readline()
        lines: list = inp.readlines()
        print('Begin analyze item pool file')
        pairs, _, _ = analyse_interaction_from_text(lines)

        item_pool: list = analyse_user_interacted_set(pairs)
        inp.close()

user_num = max(user_list + test_user_list) + 1
item_num = max(item_list + test_item_list) + 1

test_users_tensor: torch.LongTensor = torch.LongTensor(test_user_list)
test_users_tensor = test_users_tensor.to(device)
sorted_ground_truth: list = [ground_truth[user_id] for user_id in test_user_list]



#%%
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)


batch_size: int = test_batch_size
top_k_list.sort(reverse=False)
use_item_pool: bool = True


#%%
weight_dir = f"/Users/wonhyung64/Github/DRS/baselines/inv_pref/weights/expt_240222_210936_733741/epoch_1000.pt"

model = InvPrefImplicit(
    user_num=user_num,
    item_num=item_num,
    env_num=env_num,
    factor_num=factor_num,
    reg_only_embed=reg_only_embed,
    reg_env_embed=reg_env_embed
)

model.load_state_dict(torch.load(weight_dir))

model = model.to(device)

all_test_user_tensor: torch.Tensor = test_users_tensor
all_test_user_list: list = test_user_list
all_test_ground_truth: list = sorted_ground_truth

result_dicts_list: list = []

for (batch_index, (batch_users_tensor, batch_users_list, batch_users_ground_truth)) \
        in enumerate(mini_batch(batch_size, all_test_user_tensor, all_test_user_list,
                                all_test_ground_truth)):
                                

    rating_matrix: torch.Tensor = model.predict(batch_users_tensor)
    mask_users, mask_items = [], []

    for idx, user_id in enumerate(batch_users_list):
        mask_items_set = user_positive_interaction[user_id]
        mask_users += [idx] * len(mask_items_set)
        mask_items += list(mask_items_set)

    rating_matrix[mask_users, mask_items] = -(1 << 10)


    if use_item_pool:
        high_light_users, high_light_items = [], []
        for idx, user_id in enumerate(batch_users_list):
            high_light_items_set = item_pool[user_id]
            high_light_users += [idx] * len(high_light_items_set)
            high_light_items += list(high_light_items_set)
        rating_matrix[high_light_users, high_light_items] += (1 << 10)
                

            # print(rating_matrix.shape)
    _, predict_items = torch.topk(rating_matrix, k=max(top_k_list))
    predict_items: np.array = predict_items.cpu().numpy()
    # print(predict_items)
    # print(type(predict_items))
    # print(type(predict_items[0]))
    predict_items_list: list = predict_items.tolist()

    r = get_label(batch_users_ground_truth, predict_items_list)

    pre, recall, ndcg = [], [], []
    for k in top_k_list:
        ret = recall_precision_ATk(batch_users_ground_truth, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ret = NDCGatK_r(batch_users_ground_truth, r, k)
        ndcg.append(ret)

    result_dict: dict = {
        'ndcg': ndcg,
        'recall': recall,
        'precision': pre,
    }

    result_dicts_list.append(result_dict)

    # print(result_dicts_list)
result_dict: dict = merge_dict(
    dict_list=result_dicts_list,
    user_num=len(all_test_user_list)
)


#%%
users_embed_gmf: torch.Tensor = model.embed_user_invariant(batch_users_tensor)
items_embed_gmf: torch.Tensor = model.embed_item_invariant.weight

user_to_cat = []
for i in range(users_embed_gmf.shape[0]):
    tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
    tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
    user_to_cat.append(tmp)

users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

invariant_preferences: torch.Tensor = users_emb_cat * items_emb_cat
invariant_score: torch.Tensor = model.output_func(torch.sum(invariant_preferences, dim=1))

invariant_score = invariant_score.reshape(batch_users_tensor.shape[0], items_embed_gmf.shape[0])


#%%
users_embed_gmf: torch.Tensor = model.embed_user_env_aware(batch_users_tensor)
items_embed_gmf: torch.Tensor = model.embed_item_env_aware.weight


user_to_cat = []
for i in range(users_embed_gmf.shape[0]):
    tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
    tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
    user_to_cat.append(tmp)

users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

variant_preferences: torch.Tensor = users_emb_cat * items_emb_cat

#%%


        users_embed_invariant: torch.Tensor = model.embed_user_invariant(batch_users_tensor)
        # items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        items_embed_gmf: torch.Tensor = model.embed_item_invariant.weight

        users_embed_env_aware: torch.Tensor = model.embed_user_env_aware(batch_users_tensor)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim=1))
        env_aware_mid_score: torch.Tensor = self.output_func(torch.sum(env_aware_preferences, dim=1))
        env_aware_score: torch.Tensor = invariant_score * env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)
