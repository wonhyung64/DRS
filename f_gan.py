#%%
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import seaborn as sns


class ComplexGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, z):
        x = self.layer1(z)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1,1))
        self.mu = nn.Parameter(torch.ones(1))

    def forward(self, z):
        return torch.addmm(self.mu, z, self.sigma.T)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# %% Sampling from GMM
sample_size = 100000
mus = np.array([3., 6.])
sigmas = np.array([4., 2.])
pis = np.array([0.4, 0.6])

np.random.seed(0)

gmm_samples = []
for _ in range(sample_size):
    r = np.random.uniform(0, 1)

    if r >= pis[1]:
        mu = mus[1]
        sigma = sigmas[1]
    else:
        mu = mus[0]
        sigma = sigmas[0]

    sample = np.random.normal(mu, np.sqrt(sigma), 1).item()
    gmm_samples.append(sample)

    
#%% training options
batch_size = 1024
lr = 0.01
epochs = 5000


#%% train
# model_g = Generator()
model_g = ComplexGenerator()
model_g = model_g.to("mps")

model_d = Discriminator()
model_d = model_d.to("mps")

optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr)
optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr)

model_g.train()
model_d.train()
for _ in tqdm(range(epochs)):
    x = torch.tensor(np.random.choice(gmm_samples, batch_size, replace=True), dtype=torch.float32).unsqueeze(-1)
    x = x.to("mps")

    optimizer_d.zero_grad()
    z = torch.tensor(np.random.normal(0., 1., batch_size), dtype=torch.float32).unsqueeze(-1)
    z = z.to("mps")
    gen_z = model_g(z)
    real_dis = model_d(x)
    fake_dis = model_d(gen_z)
    d_loss_real = torch.mean(torch.nn.Tanh()(real_dis) * 0.5)
    d_loss_fake = torch.mean(-torch.nn.Tanh()(fake_dis) * 0.5)

    d_loss = -(d_loss_real + d_loss_fake)
    d_loss.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()
    z = torch.tensor(np.random.normal(0., 1., batch_size), dtype=torch.float32).unsqueeze(-1)
    z = z.to("mps")
    gen_z = model_g(z)
    fake_dis = model_d(gen_z)
    g_loss = -torch.mean(torch.nn.Tanh()(fake_dis) * 0.5)
    g_loss.backward()
    optimizer_g.step()


# %% histogram
z = torch.tensor(np.random.normal(0., 1., 100000), dtype=torch.float32).unsqueeze(-1).to("mps")
model_g.eval()

with torch.no_grad():
    output = model_g(z)

sns.distplot(output.cpu().numpy(), kde=True, color="red")
sns.distplot(gmm_samples, kde=True, color="blue")

# %%
