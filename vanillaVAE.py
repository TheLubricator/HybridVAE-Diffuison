# vanillaVAE.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32, stride=1)  # 64x64 -> 64x64
        self.conv2 = ConvBlock(32, 64, stride=2)  # 64x64 -> 32x32
        self.conv3 = ConvBlock(64, 128, stride=2)  # 32x32 -> 16x16
        self.conv4 = ConvBlock(128, 256, stride=2)  # 16x16 -> 8x8
        self.mu = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(256, latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.mu(x), self.logvar(x)

class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):
        super().__init__()
        self.conv1 = ConvBlock(latent_channels, 256, stride=1)  # 8x8 -> 8x8
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.conv2 = ConvBlock(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.conv3 = ConvBlock(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.conv4 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.conv1(z)
        z = self.up1(z)
        z = self.conv2(z)
        z = self.up2(z)
        z = self.conv3(z)
        z = self.up3(z)
        z = self.conv4(z)
        x = self.final(z)
        return self.sigmoid(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_loss) / x.size(0)

# New Autoencoder Classes
class AEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32, stride=1)  # 64x64 -> 64x64
        self.conv2 = ConvBlock(32, 64, stride=2)  # 64x64 -> 32x32
        self.conv3 = ConvBlock(64, 128, stride=2)  # 32x32 -> 16x16
        self.conv4 = ConvBlock(128, 256, stride=2)  # 16x16 -> 8x8
        self.latent = nn.Conv2d(256, latent_channels, kernel_size=1)  # Deterministic latent

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.latent(x)

class AEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):
        super().__init__()
        self.conv1 = ConvBlock(latent_channels, 256, stride=1)  # 8x8 -> 8x8
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.conv2 = ConvBlock(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.conv3 = ConvBlock(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.conv4 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.conv1(z)
        z = self.up1(z)
        z = self.conv2(z)
        z = self.up2(z)
        z = self.conv3(z)
        z = self.up3(z)
        z = self.conv4(z)
        x = self.final(z)
        return self.sigmoid(x)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AEEncoder()
        self.decoder = AEDecoder()

    def forward(self, x):
        z = self.encoder(x)  # Deterministic encoding
        recon = self.decoder(z)
        return recon

def ae_loss(recon_x, x):
    return nn.functional.mse_loss(recon_x, x, reduction='mean')  # Simple MSE loss