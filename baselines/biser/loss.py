import torch


def squred_loss(true, pred):
    return torch.mean(true * torch.square(1 - pred) + (1 - true) * torch.square(pred))


def l2_loss(model):
    penalty = 0.
    for param in model.parameters():
        penalty += torch.square(param).sum()
    return penalty
