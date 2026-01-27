import math
import torch
import torchvision.utils as vutils
import os


def sinusoidal_timestep_embedding(timesteps, embedding_dim: int):
    """
    Create sinusoidal embeddings for timesteps as used in many diffusion models.
    timesteps: a 1-D Tensor of shape (B,) with integer timesteps
    returns: Tensor of shape (B, embedding_dim)
    """
    # This implementation mirrors the positional encoding used in transformers
    # and the common timestep embeddings for diffusion models.
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) / (half_dim - 1)
    exponents = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * exponent
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponents.unsqueeze(0))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # pad if needed
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def save_image_grid(images, path, nrow=8):
    """Save a batch of images in a grid to `path`.
    images: tensor in range [-1,1] or [0,1], shape (B,1,28,28)
    The function will map from [-1,1] to [0,1] for visualization.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # If images are in [-1,1], convert to [0,1]
    if images.min() < -0.1:
        images = (images + 1.0) / 2.0
    vutils.save_image(images, path, nrow=nrow, normalize=False)
