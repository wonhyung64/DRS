import torch
import torch.nn as nn


class Imputator(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(Imputator, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear_2 = nn.Linear(self.embedding_k, 1, bias=False)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)

        h1 = self.linear_1(z_embed)
        h1 = torch.nn.ReLU()(h1)
        out = self.linear_2(h1)

        return out, user_embed, item_embed


class DRMse(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(DRMse, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_cvr_1 = nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear_cvr_2 = nn.Linear(self.embedding_k, 1, bias=False)

        self.linear_ctr_1 = nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear_ctr_2 = nn.Linear(self.embedding_k, 1, bias=False)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)

        cvr_h1 = self.linear_cvr_1(z_embed)
        cvr_h1 = torch.nn.ReLU()(cvr_h1)
        cvr_out = self.linear_cvr_2(cvr_h1)

        ctr_h1 = self.linear_ctr_1(z_embed)
        ctr_h1 = torch.nn.ReLU()(ctr_h1)
        ctr_out = self.linear_ctr_2(ctr_h1)

        return cvr_out, ctr_out
