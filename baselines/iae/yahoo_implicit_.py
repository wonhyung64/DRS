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

test_file = f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-test.txt'
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


# creating training data
np.zeros((num_users, num_items)).shape
all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index()
all_data = all_data.values[:, :2] + 1
unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]


#%%

from scipy import sparse


matrix = sparse.lil_matrix((num_users, num_items))
for (u, i, r) in train[:, :3]:
    matrix[int(u)-1, int(i)-1] = r
sparse_train = sparse.csr_matrix(matrix)


train_dict = {}
for idx, value in enumerate(sparse_train):
    train_dict[idx] = value.indices.copy().tolist()

batch_num = int(len(all_tr) / batch_size) +1
for b in range(batch_num):break
    batch_set_idx = all_tr[b*batch_size : (b+1)*batch_size]
    batch_matrix = np.zeros((len(batch_set_idx), num_items))
    for idx, user_id in enumerate(batch_set_idx):
        users_by_user_id = train_dict[user_id]
        batch_matrix[idx, users_by_user_id] = 1
batch_m = torch.LongTensor(batch_matrix) 
model(batch_m.type(torch.FloatTensor))

test
                model()
                feed_dict = {self.input_R: batch_matrix}
               
                _, loss = self.sess.run([self.apply_grads, self.loss], feed_dict=feed_dict)

                train_loss += loss

            ############### evaluation
            if epoch % 1 == 0:
                print(epoch,":  ", train_loss)
                weights_enc, weights_dec, bias_enc, bias_dec = \
                    self.sess.run([self.weights['encoder'], self.weights['decoder'], self.biases['encoder'], self.biases['decoder']])

                # validation
                val_ret = unbiased_evaluator(user_embed=[weights_enc, weights_dec], item_embed=[bias_enc, bias_dec], 
                                        train=self.train, val=self.val, test=self.val, num_users=self.num_users, num_items=self.num_items, 
                                        pscore=pscore, model_name=self.model_name, at_k=[3], flag_test=False, flag_unbiased = True)

                dim = self.hidden_dim

                if max_score < val_ret.loc['MAP@3', f'{self.model_name}_{dim}']:
                    max_score = val_ret.loc['MAP@3', f'{self.model_name}_{dim}']
                    print("best_val_MAP@3: ", max_score)
                    er_stop_count = 0

                    self.best_weights_enc = weights_enc
                    self.best_weights_dec = weights_dec
                    self.best_bias_enc = bias_enc
                    self.best_bias_dec = bias_dec

                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        print("stopped!")
                        break

        self.sess.close()