import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        orig_len = x.size(-1)
        
        # pad right so it's divisible by 4 (total downsampling factor)
        factor = 4
        pad_len = (factor - orig_len % factor) % factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        z = self.encoder(x)
        out = self.decoder(z)

        # crop back to original length
        out = out[..., :orig_len]
        return out


# class ConvAutoencoder1D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 500)
#             nn.ReLU(),
#             nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 250)
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), # (B, 16, 500)
#             nn.ReLU(),
#             nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),   # (B, 1, 1000)
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         out = self.decoder(z)
#         return out


class Autoencoder1D(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class VAE1D(nn.Module):
    def __init__(self, hidden_dim=32, latent_dim=8):
        super().__init__()
        # Encoder layers
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z ~ N(mu, sigma)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL divergence between posterior N(mu, σ^2) and N(0,1)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()
