import torch
import torch.nn as nn
from typing import Tuple, Dict


class DiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        num_timesteps: int,
    ):
        super().__init__()
        self.eps_model = eps_model

        for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
            self.register_buffer(name, schedule)

        self.num_timesteps = num_timesteps
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        t = torch.randint(0, self.num_timesteps, (B,), device=x.device, dtype=torch.long)

        eps = torch.randn_like(x)
        a_bar = self.alphas_cumprod[t].view(B, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        x_t = a_bar.sqrt() * x + (1.0 - a_bar).sqrt() * eps

        eps_hat = self.eps_model(x_t, t / self.num_timesteps)
        return self.criterion(eps_hat, eps)


    def sample(self, num_samples: int, size, device) -> torch.Tensor:
        B = num_samples
        z = torch.randn(B, *size, device=device)

        for i in reversed(range(self.num_timesteps)):  # i:  T-1, ..., 0
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_hat = self.eps_model(z, t / self.num_timesteps)

            a_t = self.alphas[t].view(B, 1, 1, 1)
            coef = self.one_minus_alpha_over_sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)  # = β_t / √(1-ᾱ_t)
            mean = (1.0 / a_t.sqrt()) * (z - coef * eps_hat)

            var = self.posterior_variance[t].view(B, 1, 1, 1)
            if i > 0:
                noise = torch.randn_like(z)
                z = mean + var.sqrt() * noise
            else:
                z = mean

        return z


def get_schedules(beta1: float, beta2: float, num_timesteps: int) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    betas = torch.linspace(beta1, beta2, num_timesteps, dtype=torch.float32)
    sqrt_betas = torch.sqrt(betas)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas = torch.sqrt(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

    one_minus_alpha_over_sqrt_one_minus_alpha_bar = betas / sqrt_one_minus_alphas_cumprod

    return {
        "betas": betas,
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "one_minus_alpha_over_sqrt_one_minus_alpha_bar": one_minus_alpha_over_sqrt_one_minus_alpha_bar,
    }