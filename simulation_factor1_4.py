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
        # self.bias = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1) + self.bias

        return out, user_embed, item_embed


class NonLinearity(nn.Module):
    def __init__(self, n_factors):
        super(NonLinearity, self).__init__()
        self.layer1 = nn.Linear(n_factors, n_factors)  
        self.activation1 = nn.Sigmoid()   
        self.layer2 = nn.Linear(n_factors, n_factors)
        self.activation2 = nn.Sigmoid()   
    
    def forward(self, Lambda_y, Z):
        Lambda_y = self.activation1(self.layer1(Lambda_y))
        Z = self.activation2(self.layer2(Z))
        return Lambda_y, Z


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#%%
n_factors_list = [4, 16]     # Number of latent factors
n_items_list = [20, 40, 60]      # Number of observed variables
n_samples_list = [100, 1000]  # Number of samples
treatment_effect = 1.
treat_bias = -0.5
repeat_num = 30
num_epochs = 500
batch_size = 512
lr = 1e-2
mle = torch.nn.BCELoss()
ipw = lambda x, y, z: F.binary_cross_entropy(x, y, z)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"


#%%
for n_samples in n_samples_list:
    for n_items in n_items_list:
        for n_factors in n_factors_list:

            mle_auc_list, ipw_auc_list = [], []
            for random_seed in range(1, repeat_num+1):
                # print(f"Seed {random_seed}")
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)

                # Step 1: Generate latent factors
                Z = np.random.normal(0, 1, (n_items, n_factors))  # Latent factors

                # Step 2: Define factor loading matrices for treatment and control
                Lambda_y = np.random.uniform(0., 1., (n_samples, n_factors))  # Treated group loadings
                Lambda_t = np.random.uniform(0., 1., (n_samples, n_factors))

                # Step 3: Generate treatment assignment
                prob_t_rand = np.ones([n_samples, n_items]) * 1/2
                epsilon_t = np.random.normal(0, 0.1, (n_samples, n_items))
                prob_t_real = sigmoid(Lambda_t @ Z.T + epsilon_t + treat_bias)
                T_rand = np.random.binomial(1, prob_t_rand)
                T_real = np.random.binomial(1, prob_t_real)
                # print(f"T realistic prob : {T_rand.mean()}")

                # Step 4: Generate observed variables
                epsilon_y = np.random.normal(0, 0.1, (n_samples, n_items))  # Noise for treatment

                nonlinear_Lambda_y, nonlinear_Z = NonLinearity(n_factors)(torch.Tensor(Lambda_y), torch.Tensor(Z))
                prob_y1 = sigmoid(nonlinear_Lambda_y.detach().numpy() @ nonlinear_Z.detach().numpy().T + epsilon_y + treatment_effect)  # Treatment group
                prob_y0 = sigmoid(Lambda_y @ Z.T + epsilon_y)  # Control group
                # print(prob_y1.mean(), prob_y0.std(), prob_y0.mean(), prob_y0.std())

                # Step 5: Generate binary outcome
                Y1 = np.random.binomial(1, prob_y1)
                Y0 = np.random.binomial(1, prob_y0)
                # print(f"Y1_bar : {Y1.mean()}")
                # print(f"Y0_bar : {Y0.mean()}")

                # Binary outcomes
                Y_train_ = Y1 * T_real + Y0 * (1-T_real)

                # train / test
                Y_train = Y_train_[T_real==1]
                user_idx, item_idx = np.where(T_real==1)
                x_train = np.concatenate([[user_idx],[item_idx]]).T
                ps_train = prob_t_real[T_real==1]
                num_samples = len(x_train)
                # print(f"# of observed pairs : {num_samples}")
                total_batch = num_samples // batch_size

                Y_test_ = Y1 * T_rand + Y0 * (1-T_rand)
                Y_test = Y_test_[T_rand==1]
                user_idx, item_idx = np.where(T_rand==1)
                x_test = np.concatenate([[user_idx],[item_idx]]).T


                """mle simulation"""
                model = MF(n_samples, n_items, n_factors)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        # mini-batch training
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

                    # print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

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
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        # mini-batch training
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

                    # print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

                model.eval()
                sub_x = torch.LongTensor(x_test).to(device)
                pred_, _, __ = model(sub_x)
                pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

                fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
                ipw_auc = auc(fpr, tpr)
                ipw_auc_list.append(ipw_auc)
    

            print(f"{n_samples} users, {n_items} items, {n_factors} factors")
            print(np.mean(mle_auc_list))
            print(np.std(mle_auc_list))
            print()
            print(np.mean(ipw_auc_list))
            print(np.std(ipw_auc_list))
