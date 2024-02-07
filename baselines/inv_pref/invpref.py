#%%
import os
import torch
import torch.nn as nn
import numpy as np


# %%
data_dir = "./data/coat"

os.listdir(f"{data_dir}/user_item_features")

train = np.genfromtxt(f'{data_dir}/train.ascii', encoding='ascii')
test = np.genfromtxt(f'{data_dir}/test.ascii', encoding='ascii')
propensities = np.genfromtxt(f'{data_dir}/propensities.ascii', encoding='ascii')
user_feat = np.genfromtxt(f'{data_dir}/user_item_features/user_features.ascii', encoding='ascii')
item_feat = np.genfromtxt(f'{data_dir}/user_item_features/item_features.ascii', encoding='ascii')
test.shape
train.shape

with open(f'{data_dir}/user_item_features/user_features_map.txt', "r") as f:
    user_feat_map = f.readlines()
    
with open(f'{data_dir}/user_item_features/item_features_map.txt', "r") as f:
    item_feat_map = f.readlines()


#%%
item_num = item_feat.shape[0]
user_num = user_feat.shape[0]
env_num = 4
factor_num = 30 #embedding dimesion
reg_only_embed = True
reg_env_embed = False

batch_size = 1024
epochs = 1000
cluster_interval = 30
evaluate_interval = 10
lr = 0.01

invariant_coe = 2.050646960185343
env_aware_coe = 8.632289952059462
env_coe = 5.100067503854663
L2_coe = 7.731619515414727
L1_coe = 0.0015415961377493945
alpha = 1.7379692382330174

use_class_re_weight = True
use_recommend_re_weight = True
test_begin_epoch = 0
begin_cluster_epoch = None
stop_cluster_epoch = None

cluster_use_random_sort = True

# EVALUATE_CONFIG: dict = {
    # 'eval_metric': 'mse'
# }

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]
# RANDOM_SEED_LIST = [17373331, 17373522, 17373507, 17373511, 17373423]
# RANDOM_SEED_LIST = [17373331]
# RANDOM_SEED_LIST = [999]


model = InvPrefExplicit(
    user_num, item_num, env_num, factor_num, reg_only_embed, reg_env_embed
    )

#%%
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

train_data_path = f"{data_dir}/train.csv"
test_data_path = f"{data_dir}/test.csv"

import pandas as pd
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]
test_df: pd.DataFrame = pd.read_csv(test_data_path)
train_df["user_id"].nunique()
test_df["user_id"].nunique()
train_df["item_id"].nunique()
test_df["item_id"].nunique()


_train_data: np.array = train_df.values.astype(np.int64)
_test_data: np.array = test_df.values.astype(np.int64)

_train_data_tensor: torch.Tensor = torch.LongTensor(_train_data).to(device)
_test_data_tensor: torch.Tensor = torch.LongTensor(_test_data).to(device)

user_positive_interaction = []

_user_num = int(np.max(_train_data[:, 0].reshape(-1))) + 1
_item_num = int(np.max(_train_data[:, 1].reshape(-1))) + 1

_train_pairs: np.array = _train_data[:, 0:2].astype(np.int64).reshape(-1, 2) # user_id, item_id
_test_pairs: np.array = _test_data[:, 0:2].astype(np.int64).reshape(-1, 2)

_train_pairs_tensor: torch.Tensor = torch.LongTensor(_train_pairs).to(device)
_test_pairs_tensor: torch.Tensor = torch.LongTensor(_test_pairs).to(device)

_train_scores: np.array = _train_data[:, 2].astype(np.float64).reshape(-1)
_test_scores: np.array = _test_data[:, 2].astype(np.float64).reshape(-1)

_train_scores_tensor: torch.Tensor = torch.Tensor(_train_scores).to(device)
_test_scores_tensor: torch.Tensor = torch.Tensor(_test_scores).to(device)


#%%
model
        # self.evaluator: ImplicitTestManager = evaluator
envs_num: int = model.env_num
device

training_data = torch.LongTensor(_train_data).to(device)

users_tensor: torch.Tensor = training_data[:, 0]
items_tensor: torch.Tensor = training_data[:, 1]
scores_tensor: torch.Tensor = training_data[:, 2].float()
envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, envs_num, training_data.shape[0])) # envs init
        # self.envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, 1, training_data.shape[0]))
envs = envs.to(device)
cluster_interval
evaluate_interval
batch_size
epochs
optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
recommend_loss_type = nn.MSELoss
cluster_distance_func = nn.MSELoss(reduction='none')
env_loss_type = nn.NLLLoss

invariant_coe
env_aware_coe
env_coe
L2_coe
L1_coe

epoch_cnt: int = 0
import math
batch_num = math.ceil(training_data.shape[0] / batch_size)

each_env_count = dict()

if alpha is None:
    alpha = 0.
    update_alpha = True
else:
    # alpha = alpha
    update_alpha = False

# use_class_re_weight: bool = use_class_re_weight
# use_recommend_re_weight: bool = use_recommend_re_weight
sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
class_weights: torch.Tensor = torch.Tensor(np.zeros(envs_num)).to(device)

test_begin_epoch: int = test_begin_epoch

begin_cluster_epoch: int = begin_cluster_epoch
stop_cluster_epoch: int = stop_cluster_epoch

eps_random_tensor: torch.Tensor = _init_eps(envs_num).to(device)

# cluster_use_random_sort: bool = cluster_use_random_sort

const_env_tensor_list: list = []

for env in range(envs_num):
    envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
    envs_tensor = envs_tensor.to(device)
    const_env_tensor_list.append(envs_tensor)
            

#%%
model.train()
loss_dicts_list: list = []

for (batch_index, (
        batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
)) \
        in enumerate(mini_batch(batch_size, users_tensor,
                                items_tensor, scores_tensor, envs, sample_weights)):break

    if update_alpha:
        p = float(batch_index + (epoch_cnt + 1) * batch_num) / float((epoch_cnt + 1) * batch_num)
        alpha = 2. / (1. + np.exp(-10. * p)) - 1.

    # loss_dict: dict = train_a_batch(
    #     batch_users_tensor=batch_users_tensor,
    #     batch_items_tensor=batch_items_tensor,
    #     batch_scores_tensor=batch_scores_tensor,
    #     batch_envs_tensor=batch_envs_tensor,
    #     batch_sample_weights=batch_sample_weights,
    #     alpha=alpha
    # )
    # loss_dicts_list.append(loss_dict)


        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
    invariant_score, env_aware_score, env_outputs = model(
        batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha
    )

    # print(batch_users_tensor.shape, batch_items_tensor.shape, batch_scores_tensor.shape, batch_envs_tensor.shape)
    assert batch_users_tensor.shape == batch_items_tensor.shape \
            == batch_scores_tensor.shape == batch_envs_tensor.shape
    # print(batch_users_tensor.shape, invariant_score.shape)
    assert batch_users_tensor.shape == invariant_score.shape
    # print(invariant_score.shape, env_aware_score.shape, env_outputs.shape)
    assert invariant_score.shape == env_aware_score.shape
    assert env_outputs.shape[0] == env_aware_score.shape[0] and env_outputs.shape[1] == envs_num

    if use_class_re_weight:
        env_loss = env_loss_type(reduction='none')
    else:
        env_loss = env_loss_type()

    if use_recommend_re_weight:
        recommend_loss = recommend_loss_type(reduction='none')
    else:
        recommend_loss = recommend_loss_type()

    invariant_loss: torch.Tensor = recommend_loss(invariant_score, batch_scores_tensor)
    env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)

        # print(invariant_loss, env_aware_loss, batch_sample_weights, sep='\n')

    envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

    if use_class_re_weight:
        envs_loss = torch.mean(envs_loss * batch_sample_weights)

    if use_recommend_re_weight:
        invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
        env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

    L2_reg: torch.Tensor = model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
    L1_reg: torch.Tensor = model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = invariant_loss * invariant_coe + env_aware_loss * env_aware_coe \
                             + envs_loss * env_coe + L2_reg * L2_coe + L1_reg * L1_coe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(invariant_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }


######
    epoch_cnt += 1

mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)


#%%
def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
