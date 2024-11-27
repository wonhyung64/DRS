#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GcomMF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(GcomMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = (torch.sum(user_embed.mul(item_embed), -1)).unsqueeze(-1) + self.bias

        return out, user_embed, item_embed


class ComMF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(ComMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.bias = nn.Parameter(torch.zeros(1))
        self.treatment_embedding = nn.Embedding(2, self.embedding_k//2)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        t = x[:,2]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        treatment_embed = self.treatment_embedding(t).sum([-1, -2])
        out = (torch.sum(user_embed.mul(item_embed), -1) + treatment_embed).unsqueeze(-1) + self.bias

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


def generate_total_sample(num_users, num_items):
    sample = []
    for i in range(num_users):
        sample.extend([[i,j] for j in range(num_items)])

    return np.array(sample)


#%%
# n_items_list = [20, 60]      # Number of observed variables
n_items_list = [100]      # Number of observed variables
n_factors_list = [4, 16]     # Number of latent factors
# n_samples_list = [100, 500, 1000, 5000]  # Number of samples
n_samples_list = [100]  # Number of samples
repeat_num = 30
num_epochs = 500
batch_size = 512

treatment_effect = 1.
treat_bias = -0.5
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
true_ate_dict = {}
item_result_list = []
for n_items in n_items_list:
    if n_items in true_ate_dict.keys():
        pass
    else:
        true_ate_dict[f"{n_items}"] = {}

    factor_result_list = []
    for n_factors in n_factors_list:
        np.random.seed(0)
        torch.manual_seed(0)

        # Step 1: Generate latent factors
        Z = np.random.normal(0, 1, (n_items, n_factors))  # Latent factors

        # Step 2: Define factor loading matrices for treatment and control
        Lambda_y = np.random.uniform(0., 1., (1000000, n_factors))  # Treated group loadings
        Lambda_t = np.random.uniform(0., 1., (1000000, n_factors))

        # Step 3: Generate treatment assignment

        # Step 4: Generate observed variables
        epsilon_y = np.random.normal(0, 0.1, (1000000, n_items))  # Noise for treatment

        # nonlinear_Lambda_y, nonlinear_Z = NonLinearity(n_factors)(torch.Tensor(Lambda_y), torch.Tensor(Z))
        # prob_y1 = sigmoid(nonlinear_Lambda_y.detach().numpy() @ nonlinear_Z.detach().numpy().T + epsilon_y + treatment_effect)  # Treatment group
        prob_y1 = sigmoid(Lambda_y @ Z.T + epsilon_y + treatment_effect)  # easy treatment group
        prob_y0 = sigmoid(Lambda_y @ Z.T + epsilon_y)  # Control group

        # Step 5: Generate binary outcome
        Y1 = np.random.binomial(1, prob_y1)
        Y0 = np.random.binomial(1, prob_y0)

        # TRUE ATE
        prob_t_rand = np.ones([1000000, n_items]) * 1/2
        T_rand = np.random.binomial(1, prob_t_rand)
        true_ate = Y1[T_rand==1].mean() - Y0[T_rand==0].mean()
        true_ate_dict[f"{n_items}"][f"{n_factors}"] = true_ate

        real_ate_list_n, ipw_ate_list_n, com_ate_list_n, gcom_ate_list_n = [], [], [], []
        for n_samples in n_samples_list:

            print(n_items, n_factors, n_samples)

            real_ate_list, ipw_ate_list, com_ate_list, gcom_ate_list = [], [], [], []
            for random_seed in range(1, repeat_num+1):
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)

                # Step 1: Generate latent factors
                Z = np.random.normal(0, 1, (n_items, n_factors))  # Latent factors

                # Step 2: Define factor loading matrices for treatment and control
                Lambda_y = np.random.uniform(0., 1., (n_samples, n_factors))  # Treated group loadings
                Lambda_t = np.random.uniform(0., 1., (n_samples, n_factors))

                # Step 3: Generate treatment assignment
                epsilon_t = np.random.normal(0, 0.1, (n_samples, n_items))
                prob_t_real = sigmoid(Lambda_t @ Z.T + epsilon_t + treat_bias)
                T_real = np.random.binomial(1, prob_t_real)

                # Step 4: Generate observed variables
                epsilon_y = np.random.normal(0, 0.1, (n_samples, n_items))  # Noise for treatment

                nonlinear_Lambda_y, nonlinear_Z = NonLinearity(n_factors)(torch.Tensor(Lambda_y), torch.Tensor(Z))
                prob_y1 = sigmoid(nonlinear_Lambda_y.detach().numpy() @ nonlinear_Z.detach().numpy().T + epsilon_y + treatment_effect)  # Treatment group
                prob_y0 = sigmoid(Lambda_y @ Z.T + epsilon_y)  # Control group

                # Step 5: Generate binary outcome
                Y1 = np.random.binomial(1, prob_y1)
                Y0 = np.random.binomial(1, prob_y0)
                Y_real = Y1 * T_real + Y0 * (1-T_real)

                real_ate = Y1[T_real==1].mean() - Y0[T_real==0].mean()
                real_ate_list.append(real_ate)

                ipw_ate = ((Y1[T_real==1])/(prob_t_real[T_real==1])).sum()/(n_samples*n_items) - ((Y0[T_real==0])/(1 - prob_t_real[T_real==0])).sum()/(n_samples*n_items)
                ipw_ate_list.append(ipw_ate)
                
                x_train = generate_total_sample(n_samples, n_items)
                Y_train = Y_real.flatten()
                t_train = T_real.flatten()
                num_samples = len(x_train)
                total_batch = num_samples // batch_size

                """com_ate"""
                model = ComMF(n_samples, n_items, n_factors)
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
                        sub_t = t_train[selected_idx]
                        sub_t = torch.LongTensor(sub_t).unsqueeze(-1).to(device)
                        sub_xt = torch.concat([sub_x, sub_t], -1)

                        pred, user_embed, item_embed = model(sub_xt)
                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        total_loss = rec_loss
                        epoch_total_loss += total_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model.eval()
                all_x = torch.LongTensor(x_train)
                all_t1 = torch.ones(all_x.shape[0], dtype=int).unsqueeze(-1)
                all_xt1 = torch.concat([all_x, all_t1], -1).to(device)
                all_xt0 = torch.concat([all_x, 1-all_t1], -1).to(device)

                pred_, _, __ = model(all_xt1)
                pred_y1 = nn.Sigmoid()(pred_).detach().cpu().numpy()
                pred_, _, __ = model(all_xt0)
                pred_y0 = nn.Sigmoid()(pred_).detach().cpu().numpy()
                com_ate = pred_y1.mean() - pred_y0.mean()
                com_ate_list.append(com_ate)

                "gcom_ate"
                Y_train = Y_real[T_real==1]
                user_idx, item_idx = np.where(T_real==1)
                x_train = np.concatenate([[user_idx],[item_idx]]).T
                num_samples = len(x_train)
                total_batch = num_samples // batch_size

                model1 = GcomMF(n_samples, n_items, n_factors)
                model1 = model1.to(device)
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model1.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        # mini-batch training
                        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_train[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = Y_train[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model1(sub_x)
                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        total_loss = rec_loss
                        epoch_total_loss += total_loss

                        optimizer1.zero_grad()
                        total_loss.backward()
                        optimizer1.step()

                Y_train = Y_real[T_real==0]
                user_idx, item_idx = np.where(T_real==0)
                x_train = np.concatenate([[user_idx],[item_idx]]).T
                num_samples = len(x_train)
                total_batch = num_samples // batch_size

                model0 = GcomMF(n_samples, n_items, n_factors)
                model0 = model0.to(device)
                optimizer0 = torch.optim.Adam(model0.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    all_idx = np.arange(num_samples)
                    np.random.shuffle(all_idx)
                    model1.train()

                    epoch_total_loss = 0.
                    for idx in range(total_batch):
                        # mini-batch training
                        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_train[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = Y_train[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model0(sub_x)
                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        total_loss = rec_loss
                        epoch_total_loss += total_loss

                        optimizer0.zero_grad()
                        total_loss.backward()
                        optimizer0.step()

                model1.eval()
                model0.eval()
                all_x = torch.LongTensor(x_train).to(device)
                
                pred_, _, __ = model1(all_x)
                pred_y1 = nn.Sigmoid()(pred_).detach().cpu().numpy()

                pred_, _, __ = model0(all_x)
                pred_y0 = nn.Sigmoid()(pred_).detach().cpu().numpy()

                gcom_ate = pred_y1.mean() - pred_y0.mean()
                gcom_ate_list.append(gcom_ate)


            real_ate_list_n.append([np.mean(np.array(real_ate_list) - true_ate), np.var(np.array(real_ate_list))])
            ipw_ate_list_n.append([np.mean(np.array(ipw_ate_list) - true_ate), np.var(np.array(ipw_ate_list))])
            com_ate_list_n.append([np.mean(np.array(com_ate_list) - true_ate), np.var(np.array(com_ate_list))])
            gcom_ate_list_n.append([np.mean(np.array(gcom_ate_list) - true_ate), np.var(np.array(gcom_ate_list))])


        factor_result = np.concatenate([np.array([real_ate_list_n[i], ipw_ate_list_n[i], com_ate_list_n[i], gcom_ate_list_n[i]]) for i in range(len(n_samples_list))])
        factor_result_list.append(factor_result)
    item_result = np.concatenate(factor_result_list, -1)
    item_result_list.append(item_result)

final_result = np.concatenate(item_result_list, -1) 

# %%
print(true_ate_dict)
for i in range(len(final_result)):
    print(" & ".join([f"${'{:.4f}'.format(j.round(4))}$" for j in final_result[i]]) + " \\\\")