import numpy as np
import torch
from tqdm.auto import tqdm
from monai.utils import StrEnum, unsqueeze_right
from torch.nn import functional
from torch.amp import GradScaler, autocast


class DiffusionPredictionType(StrEnum):
    """
    Set of valid prediction type names for the various scheduler's `prediction_type` argument.

    noise: predicting the noise present at x_t 
    clean: directly predicting the clean sample x_0 from any x_t
    velocity: velocity prediction, see section 2.4 https://imagen.research.google/video/paper.pdf
    flow: flow prediction, see https://diffusionflow.github.io/
    """

    NOISE = "noise"
    CLEAN = "clean"
    VELOCITY = "velocity"
    FLOW = "flow"      # v = eps - x0
    RFLOW = "rflow"    # v = x0 - eps

def reparameterize(
    x_t, α_sqrt, β_sqrt, model_output,
    prediction_type,
    clip_min=-1, clip_max=1,
):
    """
    Depending on the prediction type, reparameterize as
    the two key terms: x_0_given_t and ε_hat_t.

    `RFLOW` and `FLOW` have the same formulation with two changes:
    (i) The direction was trained differently, so we add the model output (see "whoops" above)
    (ii) The coefficients have different values (not VP)
    """

    # Get clean and noisy predictions from time t
    match prediction_type:
        case DiffusionPredictionType.NOISE:
            x_0_given_t = (x_t - β_sqrt * model_output) / α_sqrt
        case DiffusionPredictionType.CLEAN:
            x_0_given_t = model_output + (x_t - x_t.detach()) # straight-through estimation
        case DiffusionPredictionType.VELOCITY:
            x_0_given_t = α_sqrt * x_t - β_sqrt * model_output
        case DiffusionPredictionType.FLOW:
            x_0_given_t = (x_t - β_sqrt * model_output) / (α_sqrt + β_sqrt)
        case DiffusionPredictionType.RFLOW: 
            x_0_given_t = (x_t + β_sqrt * model_output) / (α_sqrt + β_sqrt)
        case _:
            raise ValueError("Invalid prediction type")

    # Clip for stability
    x_0_given_t = torch.clamp(x_0_given_t, clip_min, clip_max)

    # Compute epsilon from the CLIPPED x_0 so (x_0, epsilon) stay consistent
    ε_hat_t = (x_t - α_sqrt * x_0_given_t) / β_sqrt

    return x_0_given_t, ε_hat_t
    

def calc_loss(
    clean,
    noise,
    pred_clean,
    pred_noise,
    scheduler,
    timesteps,
    loss_type,
    loss_func=functional.mse_loss,
):

    match loss_type:
        case DiffusionPredictionType.NOISE: 
            loss = loss_func(noise, pred_noise)
        case DiffusionPredictionType.CLEAN: 
            loss = loss_func(clean, pred_clean)
        case DiffusionPredictionType.VELOCITY: 
            loss = loss_func(
                scheduler.get_velocity(clean, noise, timesteps),
                scheduler.get_velocity(pred_clean, pred_noise, timesteps),
            )
        case DiffusionPredictionType.FLOW: 
            loss = loss_func(noise - clean, pred_noise - pred_clean)
        case DiffusionPredictionType.RFLOW: # Whoops, flipped during training, have to keep flipped here
            loss = loss_func(clean - noise, pred_clean - pred_noise)
        case _:
            raise ValueError("Invalid loss type")
    return loss


class DiffusionScheduler():
    """
    Very simplified version of the MONAI scheduler class. Primary logic is in `step()`.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        set_alpha_to_one: bool = True,
        prediction_type: str = DiffusionPredictionType.FLOW,
        clip_sample_min: float = -1.0,
        clip_sample_max: float = 1.0,
        **schedule_args,
    ) -> None:
        super().__init__()

        self.prediction_type = prediction_type
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.first_alpha_cumprod = torch.tensor(0.0) if set_alpha_to_one else self.alphas_cumprod[-1]
        self.init_noise_sigma = 1.0

        if self.prediction_type == DiffusionPredictionType.RFLOW:
            self.clip_sample_min = float("-inf")
            self.clip_sample_max = float("inf")
        else:
            self.clip_sample_min = -1.0
            self.clip_sample_max = 1.0

        self.num_train_timesteps = num_train_timesteps
        self.set_timesteps(num_train_timesteps)

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1].copy().astype(np.int64)
        )
        self._schedule = timesteps.tolist()
        self._index_of = {t: i for i, t in enumerate(self._schedule)}
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def get_velocity(self, sample, noise, timesteps):
        """
        Exact copy from MONAI's monai.networks.schedulers.Scheduler.get_velocity
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, sample.ndim)
        sqrt_one_minus_alpha_prod = unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, sample.ndim
        )

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


    def sample_timesteps(self, x_start):
        """
        Randomly samples training timesteps
        """
        return torch.randint(0, self.num_train_timesteps, (x_start.shape[0],), device=x_start.device).long()

    def add_noise(self, original_samples, noise, timesteps):
        a, b = self.coefficients(timesteps.to(original_samples.device))
        a = unsqueeze_right(a, original_samples.ndim)
        b = unsqueeze_right(b, original_samples.ndim)
        return a * original_samples + b * noise

    def scheduled_neighbor(self, timestep):
        t = int(timestep)
        i = self._index_of.get(t)
        if i is None:
            spacing = self.num_train_timesteps // self.num_inference_steps
            return t - spacing
        j = i + 1
        if j < 0:
            return self.num_train_timesteps
        if j >= len(self._schedule):
            return -1
        return self._schedule[j]

    def coefficients(self, timestep):
        """(a_t, b_t) such that x_t = a_t * x_0 + b_t * eps."""

        self.betas = torch.linspace(1e-4, 2e-2, self.num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        t = torch.as_tensor(timestep, dtype=torch.float32)
        if self.prediction_type == DiffusionPredictionType.RFLOW:
            u = (t / self.num_train_timesteps).clamp(0.0, 1.0)
            return 1.0 - u, u
        idx = t.clamp(0, self.num_train_timesteps - 1).long()
        a_bar = self.alphas_cumprod.to(t.device)[idx]
        a_bar = torch.where(t < 0, self.final_alpha_cumprod.to(t.device), a_bar)
        return a_bar**0.5, (1 - a_bar)**0.5
    
    def step(self, model, t, x_t, eta=0.0, generator=None):
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

        # Final step: s = -1 gives b_s = 0, so x_s is exactly the clean estimate.
        # Returning early avoids 0 * NaN when b_t = 0 (rflow at t=0).
        if float(b_s) == 0.0:
            return x_0_given_t.detach()
    
        if eta > 0:
            σ2 = (b_s**2 / b_t**2) * (b_t**2 - (a_t * b_s / a_s)**2)
            σ_t = eta * σ2.clamp_min(0)**0.5
            n = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            x_s = a_s * x_0_given_t + (b_s**2 - σ_t**2).clamp_min(0)**0.5 * ε_hat_t + σ_t * n
        else:
            x_s = a_s * x_0_given_t + b_s * ε_hat_t
    
        return x_s.detach()
        
    def reverse_de(self, x_t, model, eta=0.0, verbose=True, **kwargs):
        """
        Run the reverse differential equation (loop through all steps)
        """
        for t in tqdm(self.timesteps, disable=not verbose):
            x_t = self.step(model, t, x_t, eta=eta, **kwargs)
        return x_t
    