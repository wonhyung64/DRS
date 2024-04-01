#%%
import codecs
import pandas as pd


#%%

# def preprocess_dataset(data: str, threshold: int = 4, alpha: float = 0.5) -> Tuple:
"""Load and Preprocess datasets."""
# load dataset.
col = {0: 'user', 1: 'item', 2: 'rate'}

train_file = f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-train.txt'
with codecs.open(f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-train.txt', 'r', 'utf-8', errors='ignore') as f:
    data_train = pd.read_csv(f, delimiter='\t', header=None)
    data_train.rename(columns=col, inplace=True)

# test_file = f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-test.txt'
with codecs.open(f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-test.txt', 'r', 'utf-8', errors='ignore') as f:
    data_test = pd.read_csv(f, delimiter='\t', header=None)
    data_test.rename(columns=col, inplace=True)

data_train.user, data_train.item = data_train.user, data_train.item
data_test.user, data_test.item = data_test.user, data_test.item


num_users, num_items = max(data_train.user.max(), data_test.user.max()), max(data_train.item.max(), data_test.item.max())

threshold = 3
# binalize rating.
data_train.rate[data_train.rate < threshold] = 0
data_train.rate[data_train.rate >= threshold] = 1
data_test.rate[data_test.rate < threshold] = 0
data_test.rate[data_test.rate >= threshold] = 1
        
print(data_train)
print(data_test)

    # train-val-test, split
train, test = data_train.values, data_test.values

import numpy as np
# train data freq
item_popularity = np.zeros(num_items, dtype=int)
for sample in train:
    if sample[2] == 1:
        item_popularity[int(sample[1]) - 1] += 1

alpha = 0.5
# for training, only tr's ratings frequency used
pscore = (item_popularity / item_popularity.max()) ** alpha

item_popularity = item_popularity**1.5 # pop^{(1+2)/2} gamma = 2


# only positive data
train = train[train[:, 2] == 1, :2]

np.zeros((num_users, num_items)).shape
# creating training data
all_data = pd.DataFrame(
    np.zeros((num_users, num_items))).stack().reset_index()
all_data = all_data.values[:, :2]
unlabeled_data = np.array(
    list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
train = np.r_[np.c_[train, np.ones(train.shape[0])],
            np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]


    # save datasets
    path_data = Path(f'./data/{data}')
    point_path = path_data / f'point_{alpha}'
    point_path.mkdir(parents=True, exist_ok=True)

    # pointwise
    np.save(file=point_path / 'train.npy', arr=train.astype(np.int))
    np.save(file=point_path / 'val.npy', arr=val.astype(np.int))
    np.save(file=point_path / 'test.npy', arr=test.astype(np.int))
    np.save(file=point_path / 'pscore.npy', arr=pscore)
    np.save(file=point_path / 'item_freq.npy', arr=item_freq)