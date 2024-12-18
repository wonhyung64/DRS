import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)


class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.sigmoid(self.linear_1(z_emb))


        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class MF_IPS_V2(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)       
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1, eta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 
        """여부시잓함"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()                      

                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)                
                pred = self.prediction_model.forward(sub_x)           
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)            
                ips_loss = xent_loss 
                             
                # ctr loss
                
                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    

                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + ips_loss +  eta * bmse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
