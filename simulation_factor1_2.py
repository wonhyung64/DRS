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
n_factors = 16     # Number of latent factors
n_items = 60      # Number of observed variables
n_samples = 1000  # Number of samples
treatment_effect = 1.
treat_bias = -0.5
repeat_num = 30
num_epochs = 1000
batch_size = 1024
mle = torch.nn.BCELoss()
ipw = lambda x, y, z: F.binary_cross_entropy(x, y, z)

mle_auc_list, ipw_auc_list = [], []
for random_seed in range(1, repeat_num+1):
# for random_seed in range(21, 51):
    print(f"Seed {random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Step 1: Generate latent factors
    Z = np.random.normal(0, 1, (n_items, n_factors))  # Latent factors

    # Step 2: Define factor loading matrices for treatment and control
    Lambda_y1 = np.random.uniform(0., 1., (n_samples, n_factors))  # Treated group loadings
    Lambda_y0 = np.random.uniform(0., 1., (n_samples, n_factors))  # Control group loadings
    Lambda_t = np.random.uniform(0., 1., (n_samples, n_factors))

    # Step 3: Generate treatment assignment
    prob_t_rand = np.ones([n_samples, n_items]) * 1/2
    epsilon_t = np.random.normal(0, 0.1, (n_samples, n_items))
    prob_t_real = sigmoid(Lambda_t @ Z.T + epsilon_t + treat_bias)
    T_rand = np.random.binomial(1, prob_t_rand)  # 50% probability of treatment
    T_real = np.random.binomial(1, prob_t_real)  # 50% probability of treatment
    print(f"T realistic prob : {T_rand.mean()}")

    # Step 4: Generate observed variables
    epsilon_y1 = np.random.normal(0, 0.1, (n_samples, n_items))  # Noise for treatment
    epsilon_y0 = np.random.normal(0, 0.1, (n_samples, n_items))  # Noise for control

    prob_y1 = sigmoid(Lambda_y1 @ Z.T + epsilon_y1 + treatment_effect)  # Treatment group
    prob_y0 = sigmoid(Lambda_y0 @ Z.T + epsilon_y0)  # Control group

    # Step 5: Generate binary outcome
    Y1 = np.random.binomial(1, prob_y1)
    Y0 = np.random.binomial(1, prob_y0)
    print(f"Y1_bar : {Y1.mean()}")
    print(f"Y0_bar : {Y0.mean()}")

    # Binary outcomes
    Y_train_ = Y1 * T_real + Y0 * (1-T_real)

    # train / test
    Y_train = Y_train_[T_real==1]
    user_idx, item_idx = np.where(T_real==1)
    x_train = np.concatenate([[user_idx],[item_idx]]).T
    ps_train = prob_t_real[T_real==1]
    num_samples = len(x_train)
    print(f"# of observed pairs : {num_samples}")
    total_batch = num_samples // batch_size

    Y_test_ = Y1 * T_rand + Y0 * (1-T_rand)
    Y_test = Y_test_[T_rand==1]
    user_idx, item_idx = np.where(T_rand==1)
    x_test = np.concatenate([[user_idx],[item_idx]]).T


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
            # mini-batch training
            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_train[selected_idx]
            sub_x = torch.LongTensor(sub_x).to("mps")
            sub_y = Y_train[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to("mps")

            pred, user_embed, item_embed = model(sub_x)

            rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
            total_loss = rec_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    model.eval()
    sub_x = torch.LongTensor(x_test).to("mps")
    pred_, _, __ = model(sub_x)
    pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
    mle_auc = auc(fpr, tpr)
    mle_auc_list.append(mle_auc)


    """ipw simulation"""
    model = MF(n_samples, n_items, n_factors)
    model = model.to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(1, num_epochs+1):
        all_idx = np.arange(num_samples)
        np.random.shuffle(all_idx)
        model.train()

        epoch_total_loss = 0.
        for idx in range(total_batch):
            # mini-batch training
            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_train[selected_idx]
            sub_x = torch.LongTensor(sub_x).to("mps")
            sub_y = Y_train[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to("mps")
            sub_ps = ps_train[selected_idx]
            sub_ps = torch.Tensor(sub_ps).to("mps")

            pred, user_embed, item_embed = model(sub_x)
            rec_loss = ipw(torch.nn.Sigmoid()(pred), sub_y, 1/sub_ps)
            total_loss = rec_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    model.eval()
    sub_x = torch.LongTensor(x_test).to("mps")
    pred_, _, __ = model(sub_x)
    pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
    ipw_auc = auc(fpr, tpr)
    ipw_auc_list.append(ipw_auc)
    

print(np.mean(mle_auc_list))
print(np.std(mle_auc_list))
print()
print(np.mean(ipw_auc_list))
print(np.std(ipw_auc_list))
print()
# %%
