import torch
import itertools
import torch.nn as nn
from torch.autograd import Function


class InvPrefExplicit(nn.Module):
    def __init__(
            self, user_num: int, item_num: int, env_num: int, factor_num: int, reg_only_embed: bool = False,
            reg_env_embed: bool = True
    ):
        super(InvPrefExplicit, self).__init__()
        self.user_num=user_num
        self.item_num=item_num
        self.env_num=env_num

        self.factor_num: int = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)

        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim=1)
        env_aware_mid_score: torch.Tensor = torch.sum(env_aware_preferences, dim=1)
        env_aware_score: torch.Tensor = invariant_score + env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def get_users_reg(self, users_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, users_id, items_id):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim=1)
        return invariant_score.reshape(-1)

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score


class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def _init_eps(envs_num):
    base_eps = 1e-10
    eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(envs_num)]
    temp: torch.Tensor = torch.Tensor(eps_list)
    eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

    return eps_random_tensor
    

class InvPrefImplicit(nn.Module):
    def __init__(
            self, user_num: int, item_num: int, env_num: int, embedding_k: int, reg_only_embed: bool = False,
            reg_env_embed: bool = True
    ):
        super(InvPrefImplicit, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num
        self.env_num: int = env_num

        self.factor_num: int = embedding_k

        self.embed_user_invariant = nn.Embedding(user_num, embedding_k)
        self.embed_item_invariant = nn.Embedding(item_num, embedding_k)

        self.embed_user_env_aware = nn.Embedding(user_num, embedding_k)
        self.embed_item_env_aware = nn.Embedding(item_num, embedding_k)

        self.embed_env = nn.Embedding(env_num, embedding_k)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(embedding_k, env_num)
        self.output_func = nn.Sigmoid()

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)

        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim=1))
        env_aware_mid_score: torch.Tensor = self.output_func(torch.sum(env_aware_preferences, dim=1))
        env_aware_score: torch.Tensor = invariant_score * env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def get_users_reg(self, users_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, x_u):
        user_id = x_u[0,0]
        items_id = x_u[:,1]
        users_embed_gmf = self.embed_user_invariant(user_id)
        items_embed_gmf = self.embed_item_invariant(items_id)
        logit = (users_embed_gmf * items_embed_gmf).sum(-1)
        return nn.Sigmoid()(logit)

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score
