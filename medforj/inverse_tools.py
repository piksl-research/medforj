import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
from .scheduler import DiffusionScheduler, reparameterize


def rician_nll(x_hat, y, sigma):
    """
    Rician NLL (up to a constant) for images in [-1, 1]. 
    First maps data to [0, 1], then computes
    sigma is in [0, 1] units.
    """
    x = ((x_hat + 1) / 2).clamp(0, 1)
    y = ((y + 1) / 2).clamp(0, 1)
    z = x * y / sigma**2
    log_i0 = torch.log(torch.special.i0e(z)) + z  # stable log I0(z), valid since z >= 0
    return (x.square() / (2 * sigma**2) - log_i0).sum()

class DPSScheduler(DiffusionScheduler):
    def __init__(self, zeta=20.0, noise_model="gaussian", sigma=None, **kwargs):
        super().__init__(**kwargs)
        self.zeta = zeta
        self.noise_model = noise_model
        self.sigma = sigma

    def step(self, model, t, x_t, y, A, eta=0.0, generator=None):
        # Make x_t differentiable for DPS guidance.
        # The model output is computed without gradients, as usual for DPS.
        x_t = x_t.detach().requires_grad_(True)

        with torch.no_grad(), autocast(device_type="cuda", enabled=x_t.is_cuda):
            model_output = model(x_t, timesteps=torch.full((x_t.shape[0],), t, device=x_t.device))
    
        s = self.scheduled_neighbor(t)

        # Diffusion scalar coefficients
        a_t, b_t = self.coefficients(t)
        a_s, b_s = self.coefficients(s if s >= 0 else -1)

        x_0_given_t, ε_hat_t = reparameterize(
            x_t, a_t, b_t, model_output, self.prediction_type,
            clip_min=self.clip_sample_min, clip_max=self.clip_sample_max,
        )

        # ===== DPS likelihood guidance =====
        if self.noise_model == 'gaussian':
            loss = torch.linalg.norm(A(x_0_given_t) - y).sum()
        elif self.noise_model == 'rician':
            loss = rician_nll(x_0_given_t, y, sigma)

        grad = torch.autograd.grad(outputs=loss, inputs=x_t)[0]
        # ===== Resume DDIM as normal =====
        
        # Final step: s = -1 gives b_s = 0, so x_s is exactly the clean estimate.
        # Returning early avoids 0 * NaN when b_t = 0 (rflow at t=0).
        if float(b_s) == 0.0:
            return x_0_given_t.detach()

        # DPS shows up here as `- self.zeta * grad` in the last line of the if/else branches
        if eta > 0:
            σ2 = (b_s**2 / b_t**2) * (b_t**2 - (a_t * b_s / a_s)**2)
            σ_t = eta * σ2.clamp_min(0)**0.5
            n = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            x_s = a_s * x_0_given_t + (b_s**2 - σ_t**2).clamp_min(0)**0.5 * ε_hat_t - self.zeta * grad + σ_t * n
        else:
            x_s = a_s * x_0_given_t + b_s * ε_hat_t - self.zeta * grad
    
        return x_s.detach()
        
    def reverse_de(self, x_t, y, A, model, eta=0.0, verbose=True, **kwargs):
        """
        Run the reverse differential equation (loop through all steps)
        """
        for t in tqdm(self.timesteps, disable=not verbose):
            x_t = self.step(model, t, x_t, y, A, eta=eta, **kwargs)
        return x_t