# latentStochasticVAE.py
import torch
import torch.nn as nn
from vanillaVAE import ConvBlock

class LSVAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):  # Updated to 4
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 16, stride=2)
        self.conv2 = ConvBlock(16, 32, stride=2)
        self.conv3 = ConvBlock(32, 64, stride=2)
        self.mu = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(64, latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.mu(x), self.logvar(x)

class LSVAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):  # Updated to 4
        super().__init__()
        self.conv1 = ConvBlock(latent_channels, 64, stride=1)  # Now expects 4 channels
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(32, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(16, 16)
        self.up3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv2d(8, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.conv1(z)
        z = self.up1(z)
        z = self.conv2(z)
        z = self.up2(z)
        z = self.conv3(z)
        z = self.up3(z)
        x = self.final(z)
        return self.sigmoid(x)

class LSVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSVAEEncoder()
        self.decoder = LSVAEDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, x, epoch=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if epoch is not None:
            noise_scale = min(0.01 * (epoch / 10), 0.015)
            z = z + noise_scale * torch.randn_like(z, device=z.device)
        recon = self.decoder(z)
        return recon, mu, logvar

def lsvae_loss(recon_x, x, mu, logvar, epoch=None, beta=0.5):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_weight = min(0.3 * (epoch / 10) if epoch is not None else 0.0, 0.3)
    return (recon_loss + kl_weight * beta * kl_loss) / x.size(0)