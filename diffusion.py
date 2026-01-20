import torch
import torch.nn.functional as F
import math
from utils import sinusoidal_timestep_embedding

# Implements the DDPM math in a compact, clear way.
# We compute a beta schedule, precompute alpha products, and provide
# functions for the forward q(x_t | x_0), computing training loss,
# and the reverse p(x_{t-1} | x_t) sampling step.


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """Simple linear schedule from beta_start to beta_end over timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPM:
    def __init__(self, timesteps=1000, device=torch.device('cpu')):
        self.timesteps = timesteps
        self.device = device
        betas = linear_beta_schedule(timesteps).to(device)

        # Precompute commonly used terms. These follow the DDPM paper.
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Useful transforms
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Posterior variance for q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _extract(self, arr, timesteps, broadcast_shape):
        """Extract values from a 1-D tensor `arr` for given batch `timesteps`.
        Then reshape to `broadcast_shape` for broadcasting in computations.
        """
        out = arr.gather(-1, timesteps).float()
        while len(out.shape) < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out.view(*broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (forward process) to timestep t: q(x_t | x_0)
        x_start: original images in [-1,1], shape (B,1,28,28)
        t: timesteps tensor (B,) with ints in [0, T-1]
        noise: optional noise to use (otherwise sampled)
        Returns: x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Equation: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_mean_variance(self, model, x_t, t):
        """
        Given x_t and model predicted noise epsilon_theta, compute the mean
        and variance of p(x_{t-1} | x_t). We follow equations from Ho et al.
        Returns: mean, variance, predicted_x0
        """
        # model predicts epsilon_theta(x_t, t)
        batch_size = x_t.shape[0]
        # Build timestep embeddings and call model
        t_emb = sinusoidal_timestep_embedding(t, 128)
        t_emb = t_emb.to(x_t.device)
        eps_theta = model(x_t, t_emb)

        # Predict x0: use analytic formula from the paper
        sqrt_recip_alphas_cumprod_t = 1.0 / self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = sqrt_recip_alphas_cumprod_t * x_t - (sqrt_recip_alphas_cumprod_t * sqrt_one_minus_alphas_cumprod_t) * eps_theta

        # Clamp predicted x0 to valid range to improve stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute posterior mean according to equation (7)-(10).
        posterior_mean_coef1 = (
            self._extract(self.betas, t, x_t.shape)
            * self._extract(self.alphas_cumprod_prev, t, x_t.shape)
            / (1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
        )
        posterior_mean_coef2 = (
            (1.0 - self._extract(self.alphas_cumprod_prev, t, x_t.shape))
            * torch.sqrt(self._extract(self.alphas, t, x_t.shape))
            / (1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
        )
        posterior_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance_t, x0_pred

    def p_sample(self, model, x_t, t):
        """Sample x_{t-1} given x_t using model's prediction."""
        mean, var, x0_pred = self.p_mean_variance(model, x_t, t)
        if (t == 0).all():
            # t==0: return mean (no noise)
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Iteratively sample starting from pure Gaussian noise x_T ~ N(0,I)
        Returns x_0 sample.
        """
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            img = self.p_sample(model, img, t)
        return img

    def loss(self, model, x_start):
        """
        Compute simple training loss for a batch: predict noise epsilon.
        Steps:
         - sample t uniformly
         - sample noise
         - compute x_t via q_sample
         - MSE between model(x_t, t_emb) and true noise
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # model prediction
        t_emb = sinusoidal_timestep_embedding(t, 128).to(x_start.device)
        eps_theta = model(x_t, t_emb)

        # MSE loss between true noise and predicted noise
        return F.mse_loss(eps_theta, noise)


if __name__ == '__main__':
    # quick sanity check
    import torch
    from model import get_model

    device = torch.device('cpu')
    ddpm = DDPM(timesteps=10, device=device)
    model = get_model(device)
    x = torch.randn(2, 1, 28, 28)
    loss = ddpm.loss(model, x)
    print('loss', loss.item())
