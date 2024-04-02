import torch.nn as nn


class UAE(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=50):
        super(UAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim #[50, 100, 200, 400]

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.latent_dim), 
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_items), 
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z
