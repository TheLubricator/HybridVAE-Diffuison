import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # For visualizations
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):  # Reusable conv block
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.conv(x)

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.features = [in_channels, 32, 64, 128]  # Progressive channels
        layers = []
        for i in range(len(self.features)-1):
            layers.append(ConvBlock(self.features[i], self.features[i+1]))
            layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)  # For 28x28 input -> 3x3 after pooling
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 3 * 3)
        self.features = [128, 64, 32, out_channels]  # Reverse
        layers = []
        for i in range(len(self.features)-1):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(ConvBlock(self.features[i], self.features[i+1]))
        self.decoder = nn.Sequential(*layers)
        self.final = nn.Sigmoid()  # For [0,1] output

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 3, 3)
        x = self.decoder(z)
        return self.final(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss