import torch
import numpy as np
import torch.nn.functional as F


def contrastive_loss(user_embed, aug_user_embed, scale=1.):
    batch_size = user_embed.shape[0]
    org_norm = F.normalize(user_embed, p=2, dim=1)
    aug_norm = F.normalize(aug_user_embed, p=2, dim=1)
    pred = F.linear(aug_norm, org_norm) / scale
    pos_label = torch.eye(batch_size).to(user_embed.device)
    neg_label = 1 - pos_label
    pos_feat = (pred.exp() * pos_label).sum(dim=-1)
    neg_feat = (pred.exp() * neg_label).sum(dim=-1)

    return -torch.log(pos_feat / (pos_feat + neg_feat)).mean()


def angle_contrastive_loss(user_embed, aug_user_embed, sim, scale=1.):
    batch_size = user_embed.shape[0]
    org_norm = F.normalize(user_embed, p=2, dim=1)
    aug_norm = F.normalize(aug_user_embed, p=2, dim=1)
    angle = torch.arccos(torch.clamp(F.linear(aug_norm, org_norm), -1+1e-7, 1-1e-7))
    pred = (angle - torch.min(sim.arccos(), angle)).cos() / scale
    pos_label = torch.eye(batch_size).to(aug_user_embed.device)
    neg_label = 1 - pos_label
    pos_feat = (pred.exp() * pos_label).sum(dim=-1)
    neg_feat = (pred.exp() * neg_label).sum(dim=-1)

    return -torch.log(pos_feat / (pos_feat + neg_feat)).mean()


def triplet_loss(anchor_user_embed, pos_user_embed, neg_user_embed, dist='sqeuclidean', margin='maxplus'):
    pos_dist = torch.square(anchor_user_embed - pos_user_embed)
    neg_dist = torch.square(anchor_user_embed - neg_user_embed)

    if dist == 'euclidean':
        pos_dist = torch.sqrt(torch.sum(pos_dist, dim=-1))
        neg_dist = torch.sqrt(torch.sum(neg_dist, dim=-1))
    elif dist == 'sqeuclidean':
        pos_dist = torch.sum(pos_dist, axis=-1)
        neg_dist = torch.sum(neg_dist, axis=-1)

    loss = pos_dist - neg_dist

    if margin == 'maxplus':
        loss = torch.maximum(torch.tensor(0.0), 1 + loss)
    elif margin == 'softplus':
        loss = torch.log(1 + torch.exp(loss))

    return torch.mean(loss)


def hard_contrastive_loss(anchor_embed, aug_embed, scale=1.):
    batch_size = anchor_embed.shape[0]
    device = anchor_embed.device
    anchor_embed = F.normalize(anchor_embed, p=2, dim=1)
    aug_embed = F.normalize(aug_embed, p=2, dim=1)
    simlarity = (anchor_embed.unsqueeze(1) * aug_embed).sum(-1) / scale
    target = torch.LongTensor(np.zeros(batch_size)).to(device)

    return torch.nn.functional.cross_entropy(simlarity, target)
