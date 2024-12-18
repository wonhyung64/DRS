#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc


class MF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)

        return out, user_embed, item_embed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%%
n_factors_list = [4, 16]
n_items_list = [20, 40, 60]
n_samples_list = [100, 1000]
treatment_effect = 1.
treat_bias = -0.5
repeat_num = 30
num_epochs = 500
batch_size = 512
mle = torch.nn.BCELoss()
ipw = lambda x, y, z: F.binary_cross_entropy(x, y, z)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"

for n_samples in n_samples_list:
    for n_items in n_items_list:
        for n_factors in n_factors_list:

            T_list, mle_auc_list, ipw_auc_list = [], [], []
            for random_seed in range(1, repeat_num+1):
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)

                Z = np.random.normal(0, 1, (n_items, n_factors))
                Lambda = np.random.uniform(0., 1., (n_samples, n_factors))

                prob_t_real = sigmoid(Lambda @ Z.T + np.random.normal(0, 0.1, (n_samples, n_items)) + treat_bias)
                T_real = np.random.binomial(1, prob_t_real)
                T_list.append(T_real.mean())

                prob_y1 = sigmoid(Lambda @ Z.T + np.random.normal(0, 0.1, (n_samples, n_items)) + treatment_effect)
                prob_y0 = sigmoid(Lambda @ Z.T + np.random.normal(0, 0.1, (n_samples, n_items)))

                Y1 = np.random.binomial(1, prob_y1)
                Y0 = np.random.binomial(1, prob_y0)
                Y = Y1 * T_real + Y0 * (1-T_real)

                Y_train = Y[T_real==1]
                user_idx, item_idx = np.where(T_real==1)
                x_train = np.concatenate([[user_idx],[item_idx]]).T
                ps_train = prob_t_real[T_real==1]
                num_samples = len(x_train)

                total_batch = num_samples // batch_size
                Y_test = Y1.reshape(-1)
                user_idx, item_idx = np.meshgrid(np.arange(n_samples), np.arange(n_items))
                x_test = np.concatenate([[user_idx, item_idx]]).T.reshape(-1, 2)

                """mle simulation"""
                model = MF(n_samples, n_items, n_factors)
                model = model.to("mps")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_train[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = Y_train[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model(sub_x)

                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        total_loss = rec_loss
                        epoch_total_loss += total_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model.eval()
                sub_x = torch.LongTensor(x_test).to(device)
                pred_, _, __ = model(sub_x)
                pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

                fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
                mle_auc = auc(fpr, tpr)
                mle_auc_list.append(mle_auc)


                """ipw simulation"""
                model = MF(n_samples, n_items, n_factors)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_train[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = Y_train[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                        sub_ps = ps_train[selected_idx]
                        sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model(sub_x)
                        rec_loss = ipw(torch.nn.Sigmoid()(pred), sub_y, 1/sub_ps)
                        total_loss = rec_loss
                        epoch_total_loss += total_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model.eval()
                sub_x = torch.LongTensor(x_test).to(device)
                pred_, _, __ = model(sub_x)
                pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

                fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
                ipw_auc = auc(fpr, tpr)
                ipw_auc_list.append(ipw_auc)

            print(f"{n_samples} users, {n_items} items, {n_factors} factors")
            print(f"T_bar : {np.mean(T_list)}")
            print(np.mean(mle_auc_list))
            print(np.std(mle_auc_list))
            print(np.mean(ipw_auc_list))
            print(np.std(ipw_auc_list))
            print()
