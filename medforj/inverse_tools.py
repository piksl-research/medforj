import torch
import math
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
from .scheduler import DiffusionScheduler, reparameterize


class DPSScheduler(DiffusionScheduler):
    def __init__(self, zeta=20.0, **kwargs):
        super().__init__(**kwargs)
        self.zeta = zeta

    def step(self, px, t, x_t, y, forward_model, step_index, eta=0.0, generator=None):
        x_t = x_t.detach().requires_grad_(True)
        with torch.no_grad(), autocast(device_type="cuda", enabled=x_t.is_cuda):
            px_output = px(x_t, timesteps=t.to(x_t.device).repeat(x_t.shape[0]))

        s = self.next_timesteps[step_index]
        a_t, b_t = self.coefficients(t)
        a_s, b_s = self.coefficients(s)
        
        x_0_given_t, ε_hat_t = reparameterize(
            x_t, a_t, b_t, px_output, self.prediction_type,
            clip_min=self.clip_sample_min, clip_max=self.clip_sample_max,
        )

        # ===== DPS likelihood guidance =====
        loss = forward_model.loss(x_0_given_t, y)
        grad = torch.autograd.grad(outputs=loss, inputs=x_t)[0]
        # ===== Resume DDIM as normal =====
        
        # Final step: s = -1 gives b_s = 0, so x_s is exactly the clean estimate.
        # Returning early avoids 0 * NaN when b_t = 0 (rflow at t=0).
        if float(b_s) == 0.0:
            return (x_0_given_t - self.zeta * grad).detach()

        # DPS shows up here as `- self.zeta * grad` in the last line of the if/else branches
        if eta > 0:
            σ2 = (b_s**2 / b_t**2) * (b_t**2 - (a_t * b_s / a_s)**2)
            σ_t = eta * σ2.clamp_min(0)**0.5
            n = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            x_s = a_s * x_0_given_t + (b_s**2 - σ_t**2).clamp_min(0)**0.5 * ε_hat_t - self.zeta * grad + σ_t * n
        else:
            x_s = a_s * x_0_given_t + b_s * ε_hat_t - self.zeta * grad
    
        return x_s.detach()
        
    def reverse_de(self, x_t, y, forward_model, px, eta=0.0, verbose=True, **kwargs):
        for i, t in enumerate(tqdm(self.timesteps, disable=not verbose)):
            x_t = self.step(px, t, x_t, y, forward_model, step_index=i, eta=eta, **kwargs)
        return x_t

def _batch_dot(a, b):
    """Per-sample inner product, shape (B,)."""
    return (a.flatten(start_dim=1) * b.flatten(start_dim=1)).sum(dim=1)


def _dc_loss(Ax, y, noise_model, sigma):
    """
    Data-consistency loss for the gradient-descent route. 
        squared L2 for Gaussian tasks
        squared L2 scaled by 1 / (2 sigma^2) for Rician denoising. 
    Note that this is NOT the Rician NLL (or the unsquared L2) used by DPSScheduler.
    """
    sq = (Ax - y).square()
    if noise_model == "rician":
        sq = sq / (2.0 * sigma**2)
    return sq.flatten(start_dim=1).sum(dim=1).mean()


class ReSampleScheduler(DiffusionScheduler):
    """
    ReSample (Song et al., 2023) for the latent rectified-flow prior.

    Latent convention (MONAI RFlow):
        z_τ = (1 - τ) z_0 + τ ε,   px predicts v = z_0 - ε,   τ = t / T.

    Every step takes an unconditional RF step from t to s. At resampling events,
    the clean latent estimate is decoded, made data-consistent in pixel space
    (in [-1, 1], the same convention as the voxel-space operators), re-encoded,
    and stochastically resampled back onto the trajectory at time s.

    Pixel-space data consistency depends on the operator:
      - A has an adjoint (slice selection, inpainting): exact proximal solve
            (AᵀA + λI) x = Aᵀy + λ x_init   by conjugate gradient.
      - otherwise (k-space, motion, Rician): AdamW on the data loss through A.
    """

    def __init__(
        self,
        ae,
        resample_events=8,
        gamma=40.0,
        cg_lam=1e-2,
        cg_iters=20,
        cg_tol=1e-6,
        gd_lr=1e-2,
        gd_iters=100,
        gd_tol=1e-4,
        **kwargs,
    ):
        super().__init__(prediction_type="rflow", **kwargs)
        self.ae = ae
        self.resample_events = resample_events
        self.gamma = float(gamma)
        self.cg_lam = float(cg_lam)
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.gd_lr = gd_lr
        self.gd_iters = gd_iters
        self.gd_tol = gd_tol

    def is_resample_step(self, step_index):
        # Fixed event COUNT rather than a fixed stride, so the number of lossy
        # encode/decode round-trips does not grow with the step count. The final
        # step is always an event too (resample_events + 1 in total).
        every = max(1, self.num_inference_steps // max(1, self.resample_events))
        return step_index % every == 0 or step_index == self.num_inference_steps - 1

    def step(self, px, t, z_t, y, forward_model, step_index, generator=None):
        N, T = self.num_inference_steps, self.num_train_timesteps
        s = self.next_timesteps[step_index]
        τ_t = min(max(float(t) / T, 0.0), 1.0)
        τ_s = min(max(float(s) / T, 0.0), 1.0)

        z_t = z_t.detach()
        with torch.no_grad(), autocast(device_type="cuda", enabled=z_t.is_cuda):
            v_hat = px(z_t, timesteps=torch.full((z_t.shape[0],), float(t), device=z_t.device))

        # RF clean/noise estimates, then the unconditional step to s
        z_0_given_t = z_t + τ_t * v_hat
        ε_hat_t = z_t - (1.0 - τ_t) * v_hat
        z_s = (1.0 - τ_s) * z_0_given_t + τ_s * ε_hat_t

        if not self.is_resample_step(step_index):
            return z_s.detach()

        # ===== ReSample: hard data consistency, then stochastic resampling =====
        z_0_y = self.data_consistency(z_0_given_t, y, forward_model)
        return self.stochastic_resample(z_0_y, z_s, τ_s, generator).detach()

    def data_consistency(self, z_0, y, forward_model):
        """z_0 -> decode -> pixel-space DC in [-1, 1] -> encode. No decoder backprop."""
        with torch.no_grad(), autocast(device_type="cuda", enabled=z_0.is_cuda):
            x_init = (self.ae.decode_stage_2_outputs(z_0) * 2.0 - 1.0).clamp(-1.0, 1.0)

        # DC runs in fp32: the CG dot products and the summed loss overflow fp16,
        # and cuFFT rejects non-power-of-2 sizes in half precision.
        x_init, y = x_init.float(), y.float()
        if forward_model.A.has_adjoint:
            x_y = self.prox_cg(x_init, y, forward_model.A)
        else:
            x_y = self.gd(x_init, y, forward_model)

        with torch.no_grad(), autocast(device_type="cuda", enabled=x_y.is_cuda):
            return self.ae.encode_stage_2_inputs((x_y + 1.0) / 2.0)

    @torch.no_grad()
    def prox_cg(self, x_init, y, A):
        """
        x_y = argmin_x ½‖Ax - y‖² + ½λ‖x - x_init‖², by CG on the normal equations
        (AᵀA + λI) x = Aᵀy + λ x_init, warm-started at x_init.
        """
        λ = self.cg_lam
        normal_op = lambda x: A.adjoint(A(x)) + λ * x
        view = (-1,) + (1,) * (x_init.ndim - 1)

        with autocast(device_type="cuda", enabled=False):
            b = A.adjoint(y) + λ * x_init
            x = x_init.clone()
            r = b - normal_op(x)
            p = r.clone()
            rs = _batch_dot(r, r)
            for _ in range(self.cg_iters):
                Ap = normal_op(p)
                α = (rs / _batch_dot(p, Ap).clamp_min(1e-12)).view(view)
                x = x + α * p
                r = r - α * Ap
                rs_new = _batch_dot(r, r)
                if rs_new.max().sqrt().item() < self.cg_tol:
                    break
                p = r + (rs_new / rs.clamp_min(1e-12)).view(view) * p
                rs = rs_new
        return x.clamp(-1.0, 1.0)

    def gd(self, x_init, y, forward_model):
        """AdamW on the data loss through A, projecting onto [-1, 1] after each step."""
        x = x_init.clone().requires_grad_(True)
        opt = torch.optim.AdamW([x], lr=self.gd_lr)  # default weight_decay=0.01
        with torch.enable_grad(), autocast(device_type="cuda", enabled=False):
            for _ in range(self.gd_iters):
                opt.zero_grad(set_to_none=True)
                loss = _dc_loss(forward_model.A(x), y, forward_model.noise_model, forward_model.sigma)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-1.0, 1.0)
                if loss.item() <= self.gd_tol:
                    break
        return x.detach()

    def stochastic_resample(self, z_0_y, z_s_prime, τ_s, generator=None):
        """
        ReSample's stochastic resampling in RF form: the Gaussian product of
        N((1 - τ_s) z_0_y, τ_s² I) and N(z'_s, γ τ_s² I). With γ = 40 the result
        leans heavily (40/41) towards the data-consistent estimate.
        """
        q = max(τ_s**2, 1e-8)
        σ2 = self.gamma * q
        denom = σ2 + q
        mean = (σ2 * (1.0 - τ_s) * z_0_y + q * z_s_prime) / denom
        var = σ2 * q / denom
        if τ_s > 0 and var > 0:
            n = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            mean = mean + math.sqrt(var) * n
        return mean

    def reverse_de(self, z_t, y, forward_model, px, eta=0.0, verbose=True, **kwargs):
        """
        Returns the LATENT; map to image space with `decode`. `eta` is unused
        (ReSample's stochasticity lives in the resampling step) and is accepted
        only so the call matches DPSScheduler.reverse_de.
        """
        for i, t in enumerate(tqdm(self.timesteps, disable=not verbose)):
            z_t = self.step(px, t, z_t, y, forward_model, step_index=i, **kwargs)
        return z_t

    @torch.no_grad()
    def decode(self, z):
        """Latent -> image in [-1, 1]."""
        with autocast(device_type="cuda", enabled=z.is_cuda):
            x = self.ae.decode_stage_2_outputs(z)
        return x