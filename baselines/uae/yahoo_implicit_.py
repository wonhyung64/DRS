#%%
import codecs
import pandas as pd


#%%

# def preprocess_dataset(data: str, threshold: int = 4, alpha: float = 0.5) -> Tuple:
"""Load and Preprocess datasets."""
# load dataset.
col = {0: 'user', 1: 'item', 2: 'rate'}

# train_file = f'/Users/wonhyung64/Github/DRS/data/yahoo_r3/ydata-ymusic-rating-study-v1_0-train.txt'
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


# creating training data
np.zeros((num_users, num_items)).shape
all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index()
all_data = all_data.values[:, :2] + 1
unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]


#%%
from tqdm import tqdm
random_state = 0
"""
Add methods in AE models 
"""
sub_results_sum = pd.DataFrame()

# train
if self.model_name in ['uae', 'iae']:
    weights_enc, weights_dec, bias_enc, bias_dec = ae_trainer(sess, data=self.data, train=train, val=val, test=test,
                num_users=num_users, num_items=num_items, n_components=self.dim, 
                eta=self.eta, lam=self.lam, max_iters=self.max_iters, batch_size=self.batch_size, 
                model_name=self.model_name, item_freq=item_freq,
                unbiased_eval = self.unbiased_eval, random_state=random_state)

def ae_trainer(sess: tf.Session, data, train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users: int, num_items: int,
                   n_components: int, eta, lam: float, max_iters, batch_size,
                   model_name: str, item_freq: np.ndarray,
                   unbiased_eval: bool, random_state: int, wu=0.1, wi=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Train autoencoder models."""
    if model_name == 'uae':
        model = uAE(sess, data,  train, val, test, num_users, num_items, hidden_dim=n_components, eta=eta, random_state=random_state,
                    reg=lam, max_iters=max_iters, batch_size=batch_size)
        model.train_model(pscore=item_freq, unbiased_eval=unbiased_eval)

import torch.nn as nn
class UAE(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, latent_dim=50):
        super(UAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim #[50, 100, 200, 400]

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.latent_dim), 
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_items), 
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z



model = UAE(num_users, num_items)
# model = model.to(device)

import torch
def squred_loss(true, pred):
    return torch.mean(true * torch.square(1 - pred) + (1 - true) * torch.square(pred))

def l2_loss(model):
    penalty = 0.
    for param in model.parameters():
        penalty += torch.square(param).sum()
    return penalty

l2_lambda = 0.00001
lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


class uAE(AbstractRecommender):
    def __init__(self, sess, data, train, val, test, num_user: np.array, num_item: np.array, \
                 hidden_dim: int, eta: float, reg: float, max_iters: int, batch_size: int, random_state: int) -> None:        
        """Initialize Class."""
        self.data = data
        self.num_users = num_user
        self.num_items = num_item
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.reg = reg
        self.num_epochs = max_iters
        self.batch_size = batch_size
        self.train = train
        self.val = val
        self.test = test
        self.train_dict = csr_to_user_dict(tocsr(train, num_user, num_item))
        self.sess = sess

        self.model_name = 'uae'
        self.random_state = random_state

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

        self.best_weights_enc = None
        self.best_weights_dec = None
        self.best_bias_enc = None
        self.best_bias_dec = None

    def create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_R = tf.placeholder(tf.float32, [None, self.num_items])

            self.tmp1 = tf.placeholder(tf.float32, [None, self.num_items])
            self.tmp2 = tf.placeholder(tf.float32, [None, self.num_items])
            self.tmp3 = tf.placeholder(tf.float32, [None, self.num_items])

    def build_graph(self):
        with tf.name_scope("embedding_layer"):  # The embedding initialization is unknown now
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_state)
             
            self.weights = {'encoder': tf.Variable(initializer([self.num_items, self.hidden_dim])),
                            'decoder': tf.Variable(initializer([self.hidden_dim, self.num_items]))}
            self.biases = {'encoder': tf.Variable(initializer([self.hidden_dim])),
                           'decoder': tf.Variable(initializer([self.num_items]))}

        with tf.name_scope("prediction"):
            input_R = self.input_R
            self.encoder_op = tf.sigmoid(tf.matmul(input_R, self.weights['encoder']) +
                                                  self.biases['encoder'])
            
            self.decoder_op = tf.matmul(self.encoder_op, self.weights['decoder']) + self.biases['decoder']
            self.output = tf.sigmoid(self.decoder_op)


    def create_losses(self):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum( self.input_R * tf.square(1. - self.output) + (1 - self.input_R) * tf.square(self.output) )
            
            self.reg_loss = self.reg*l2_loss(self.weights['encoder'], self.weights['decoder'],
                                             self.biases['encoder'], self.biases['decoder'])

            self.loss = self.loss + self.reg_loss


    def add_optimizer(self):
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdagradOptimizer(learning_rate=self.eta).minimize(self.loss)

    def train_model(self, pscore, unbiased_eval):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        max_score = 0
        er_stop_count = 0
        early_stop = 5

        all_tr = np.arange(self.num_users)
        for epoch in range(self.num_epochs):
            train_loss = 0

            np.random.RandomState(12345).shuffle(all_tr)

            batch_num = int(len(all_tr) / self.batch_size) +1
            for b in range(batch_num):
                batch_set_idx = all_tr[b*self.batch_size : (b+1)*self.batch_size]
                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                for idx, user_id in enumerate(batch_set_idx):
                    users_by_user_id = self.train_dict[user_id]
                    batch_matrix[idx, users_by_user_id] = 1
    
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