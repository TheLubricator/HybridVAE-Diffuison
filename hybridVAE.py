# hybridVAE.py
import torch
import torch.nn as nn
from vanillaVAE import *  # Import baseline VAE

def linear_beta_schedule(timesteps=1000):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

alphas = 1.0 - linear_beta_schedule()
alphas_cumprod = torch.cumprod(alphas, dim=0)

def forward_diffusion(z, t, noise=None, device='cpu'):
    alphas_cumprod_device = alphas_cumprod.to(z.device)
    if noise is None:
        noise = torch.randn_like(z, device=z.device)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_device[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod_device[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise, noise

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

class UNet(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.time_emb = nn.Embedding(1000, latent_channels)  # Timestep embedding
        self.down1 = ConvBlock(latent_channels, 64)  # 8x8 -> 8x8
        self.pool1 = nn.MaxPool2d(2)  # 8x8 -> 4x4
        self.down2 = ConvBlock(64, 128)  # 4x4 -> 4x4
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.up2 = ConvBlock(64 + 64, 64)  # Skip connection
        self.final = nn.Conv2d(64, latent_channels, kernel_size=1)  # Predict noise

    def forward(self, x, t):
        t_emb = self.time_emb(t)[:, :, None, None]  # (batch, channels, 1, 1)
        x = x + t_emb
        d1 = self.down1(x)
        d2 = self.pool1(d1)
        d3 = self.down2(d2)
        u1 = self.up1(d3)
        u1 = torch.cat([u1, d1], dim=1)  # Skip
        u2 = self.up2(u1)
        return self.final(u2)

def diffusion_loss(model, z, timesteps=1000, device='cpu'):
    t = torch.randint(0, timesteps, (z.shape[0],)).to(device)
    noisy_z, noise = forward_diffusion(z, t, device=device)
    pred_noise = model(noisy_z, t)
    return nn.functional.mse_loss(pred_noise, noise)

def consistency_loss(vae, z, num_samples=5, device='cpu'):
    variants = []
    for _ in range(num_samples):
        recon = vae.decoder(vae.reparameterize(*vae.encoder(vae.decoder(z))))
        variants.append(recon)
    var = torch.var(torch.stack(variants), dim=0).mean()
    return var

class HybridVAEDiffusion(nn.Module):
    def __init__(self, vae, timesteps=1000):
        super().__init__()
        self.vae = vae  # Use the updated 1-channel VAE
        self.unet = UNet()
        self.timesteps = timesteps
        self.alphas = alphas.to(vae.encoder.conv1.conv[0].weight.device)  # Device consistency
        self.alphas_cumprod = alphas_cumprod.to(vae.encoder.conv1.conv[0].weight.device)

    def forward(self, x):
        with torch.no_grad():
            mu, logvar = self.vae.encoder(x)  # x is [batch, 1, 64, 64]
            z = self.vae.reparameterize(mu, logvar)  # z is [batch, 4, 8, 8]
        # Diffusion process would go here (e.g., training UNet)
        # For now, return baseline reconstruction
        return self.vae.decoder(z)

    def generate(self, z, steps=1000, device=None):
        # Ensure device matches z
        if device is None:
            device = z.device
        for t in range(steps - 1, -1, -1):
            t_tensor = torch.full((z.shape[0],), t, dtype=torch.long, device=z.device)
            noisy_z, _ = forward_diffusion(z, t_tensor, device=z.device)
            pred_noise = self.unet(noisy_z, t_tensor)
            z = (noisy_z - pred_noise) / torch.sqrt(1. - alphas[t]).to(z.device)  # Ensure same device
        return self.vae.decoder(z)