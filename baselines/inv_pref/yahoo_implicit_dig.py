#%%
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import InvPrefImplicit


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

    inp.close()

with open(test_data_path, 'r') as inp:
    inp.readline()
    lines: list = inp.readlines()
    print('Begin analyze raw test file')
    pairs, test_user_list, test_item_list = analyse_interaction_from_text(lines)
    ground_truth: list = analyse_user_interacted_set(pairs)
    inp.close()

user_num = max(user_list + test_user_list) + 1
item_num = max(item_list + test_item_list) + 1


#%%
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)


batch_size: int = test_batch_size
top_k_list.sort(reverse=False)
use_item_pool: bool = True


#%%
checkpoint_dir = f"/Users/wonhyung64/Github/DRS/baselines/inv_pref/weights/expt_240226_122140_943845"
epoch_num = 100

model = InvPrefImplicit(
    user_num=user_num,
    item_num=item_num,
    env_num=env_num,
    factor_num=factor_num,
    reg_only_embed=reg_only_embed,
    reg_env_embed=reg_env_embed
)
weight_dir = f"{checkpoint_dir}/epoch_{epoch_num}.pt"
model.load_state_dict(torch.load(weight_dir))

model = model.to(device)

train_tensor: torch.LongTensor = torch.LongTensor(_train_data).to(device)

assert train_tensor.shape[1] == 3

envs_num: int = model.env_num
users_tensor: torch.Tensor = train_tensor[:, 0]
items_tensor: torch.Tensor = train_tensor[:, 1]
scores_tensor: torch.Tensor = train_tensor[:, 2].float()

envs_ = np.load(f"{checkpoint_dir}/env_epoch_{epoch_num}.npy")
envs: torch.LongTensor = torch.LongTensor(envs_)
envs = envs.to(device)


#%%
inv_prefs = []
var_prefs = []
total_loss = []

model.eval()
for (batch_index, (
        batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor
)) \
        in tqdm(enumerate(mini_batch(batch_size, users_tensor,
                                items_tensor, scores_tensor, envs))):

    with torch.no_grad():
        users_embed_invariant: torch.Tensor = model.embed_user_invariant(batch_users_tensor)
        items_embed_invariant: torch.Tensor = model.embed_item_invariant(batch_items_tensor)

        users_embed_env_aware: torch.Tensor = model.embed_user_env_aware(batch_users_tensor)
        items_embed_env_aware: torch.Tensor = model.embed_item_env_aware(batch_items_tensor)

        envs_embed: torch.Tensor = model.embed_env(batch_envs_tensor)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        env_outputs: torch.Tensor = model.env_classifier(env_aware_preferences)
        env_outputs = env_outputs.reshape(-1, env_num)

    env_outputs = env_outputs.to("cpu")
    batch_envs_tensor = batch_envs_tensor.to("cpu")
    env_loss = nn.NLLLoss(reduction="none")(env_outputs, batch_envs_tensor)

    inv_prefs.append(invariant_preferences)
    var_prefs.append(env_aware_preferences)
    total_loss.append(env_loss)

inv_prefs = torch.concat(inv_prefs, dim=0).cpu().numpy()
var_prefs = torch.concat(var_prefs, dim=0).cpu().numpy()
total_loss = torch.concat(total_loss, dim=0).cpu().numpy()


os.makedirs(f"{checkpoint_dir}/inv_prefs", exist_ok=True)
os.makedirs(f"{checkpoint_dir}/var_prefs", exist_ok=True)
os.makedirs(f"{checkpoint_dir}/env_loss", exist_ok=True)

np.save(f"{checkpoint_dir}/inv_prefs/e{epoch_num}.npy", inv_prefs, allow_pickle=True)
np.save(f"{checkpoint_dir}/var_prefs/e{epoch_num}.npy", var_prefs, allow_pickle=True)
np.save(f"{checkpoint_dir}/env_loss/e{epoch_num}.npy", total_loss, allow_pickle=True)

# %%
env_losses = []
for i in range(100, 1100, 100):
    env_loss = np.load(f"{checkpoint_dir}/env_loss/e{i}.npy")
    env_losses.append(env_loss.mean())

plt.plot(range(100,1100,100), env_losses)
plt.xlabel("Epochs")
plt.ylabel("NLL loss")


#%% t-SNE
sample_num = 2000
seed = 0

inv_pref = np.load(f"{checkpoint_dir}/inv_prefs/e{epoch_num}.npy")
var_pref = np.load(f"{checkpoint_dir}/var_prefs/e{epoch_num}.npy")

np.random.seed(seed)
indices = np.random.choice(inv_prefs.shape[0], sample_num)
inv_sample = inv_pref[indices]
var_sample = var_pref[indices]
envs_sample = envs[indices]

colors = ["blue"]*sample_num + ["orange"]*sample_num
# colors = ["blue"]*sample_num + ["orange" if env else "red" for env in envs_sample]
# colors = ["blue" if env else "green" for env in envs_sample] + ["orange" if env else "red" for env in envs_sample]

prefs = np.concatenate([inv_sample, var_sample], axis=0)

model = TSNE(n_components=3)
reducted_prefs = model.fit_transform(prefs)

epoch_tsne.append([reducted_prefs, colors])
#%%2d


# %%3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(reducted_prefs[:,0], reducted_prefs[:,1], reducted_prefs[:,2], c=colors)
ax.ticks("on")


# %%
titles = ["Initial", "Epoch 10", "Epoch 50", "Epoch 100"]
fig = plt.figure(figsize=plt.figaspect(0.25))
for i, (reducted_prefs, colors) in enumerate(epoch_tsne):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    ax.scatter(reducted_prefs[:,0], reducted_prefs[:,1], reducted_prefs[:,2], c=colors)
    ax.title.set_text(f"{titles[i]}")
    

# %%
