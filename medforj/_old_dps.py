import torch
import torch.nn.functional as F
import math
import numpy as np
from monai.utils import StrEnum
from monai.networks.schedulers import Scheduler
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from typing import Literal, Optional, Sequence, Callable

from .scheduler import scheduled_neighbor



def _maybe_empty_cache(enabled: bool = True):
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_module(module: torch.nn.Module):
    """
    Freeze parameters but still allow autograd through module inputs if used elsewhere.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def freeze_modules(*modules):
    for module in modules:
        if isinstance(module, torch.nn.Module):
            freeze_module(module)


def memory_report(tag: str = ""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] allocated={alloc:.2f} GB | reserved={reserved:.2f} GB | peak={peak:.2f} GB")


# ===== Some code for Rician noise distributions =====
def _expand_batch_timesteps(timestep, x: torch.Tensor) -> torch.Tensor:
    """
    Creates a timestep tensor of shape (batch,) on x.device.
    """
    if torch.is_tensor(timestep):
        timestep = timestep.to(device=x.device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(x.shape[0])
        return timestep

    return torch.full(
        (x.shape[0],),
        float(timestep),
        device=x.device,
        dtype=torch.float32,
    )


def log_i0_stable(z: torch.Tensor) -> torch.Tensor:
    """
    Stable log(I0(z)).

    For Rician likelihood:
        log I0(z), z >= 0.

    torch.special.i0e(z) = exp(-abs(z)) I0(z), so
        log I0(z) = log i0e(z) + abs(z).
    """
    z = torch.clamp(z, min=0.0)

    if hasattr(torch.special, "i0e"):
        return torch.log(torch.special.i0e(z).clamp_min(1e-30)) + z

    return torch.log(torch.special.i0(z).clamp_min(1e-30))


def to_unit_interval(x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Maps [-1, 1] to [0, 1].
    """
    x01 = 0.5 * (x + 1.0)
    if clamp:
        x01 = x01.clamp(0.0, 1.0)
    return x01


def gaussian_data_loss(
    recon: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "batch_sum",
    sigma = None, # Unused arg
) -> torch.Tensor:
    """
    Gaussian negative log likelihood up to constants.

    If sigma is provided, uses:
        ||recon - y||^2 / (2 sigma^2)

    If sigma is None, uses unscaled squared error.
    """

    loss = torch.linalg.norm(recon - y).sum()

    if reduction == "batch_sum":
        return loss.sum()
    if reduction == "batch_mean":
        return loss.mean()
    if reduction == "none":
        return loss

    raise ValueError(f"Unknown reduction: {reduction}")


def rician_nll_from_minus1_1(
    recon: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    reduction: str = "batch_sum",
    clamp_recon: bool = True,
    clamp_y: bool = True,
) -> torch.Tensor:
    """
    Rician negative log likelihood for magnitude images represented in [-1, 1].

    Internally maps recon and y to [0, 1]:

        recon01 = (recon + 1) / 2
        y01     = (y + 1) / 2

    The Rician density is:

        p(y | x) = y / sigma^2
                  exp(-(y^2 + x^2) / (2 sigma^2))
                  I0(y x / sigma^2)

    Dropping terms independent of x, the NLL is:

        x^2 / (2 sigma^2) - log I0(y x / sigma^2)

    sigma must be measured in [0, 1] intensity units.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive for Rician likelihood.")

    recon01 = to_unit_interval(recon, clamp=clamp_recon)
    y01 = to_unit_interval(y, clamp=clamp_y)

    sigma2 = sigma**2
    z = y01 * recon01 / sigma2

    nll = recon01.square() / (2.0 * sigma2) - log_i0_stable(z)
    loss_per_sample = nll.reshape(nll.shape[0], -1).sum(dim=1)

    if reduction == "batch_sum":
        return loss_per_sample.sum()
    if reduction == "batch_mean":
        return loss_per_sample.mean()
    if reduction == "none":
        return loss_per_sample

    raise ValueError(f"Unknown reduction: {reduction}")


def measurement_loss(
    recon: torch.Tensor,
    y: torch.Tensor,
    noise_model: str,
    sigma: float | None,
    reduction: str = "batch_sum",
) -> torch.Tensor:
    """
    Dispatches to the requested measurement likelihood.

    noise_model:
        "gaussian": Gaussian NLL / squared-error data term.
        "rician": Rician NLL for magnitude data stored in [-1, 1].
    """
    noise_model = noise_model.lower()

    if noise_model in {"gaussian", "l2", "mse"}:
        return gaussian_data_loss(
            recon=recon,
            y=y,
            sigma=sigma,
            reduction=reduction,
        )

    if noise_model in {"rician", "rice"}:
        if sigma is None:
            raise ValueError("sigma must be provided for Rician likelihood.")
        return rician_nll_from_minus1_1(
            recon=recon,
            y=y,
            sigma=sigma,
            reduction=reduction,
            clamp_recon=True,
            clamp_y=True,
        )

    raise ValueError(
        f"Unknown noise_model={noise_model!r}. "
        "Expected one of: 'gaussian', 'rician'."
    )


# ===== Some stuff from usual Schedulers in medforj =====

class DDIMPredictionType(StrEnum):
    """
    Set of valid prediction type names for the DDIM scheduler's `prediction_type` argument.

    epsilon: predicting the noise of the diffusion process
    sample: directly predicting the noisy sample
    velocity: velocity prediction, see section 2.4 https://imagen.research.google/video/paper.pdf
    flow: flow prediction, see https://diffusionflow.github.io/
    """

    EPSILON = "epsilon"
    SAMPLE = "sample"
    VELOCITY = "velocity"
    FLOW = "flow"
    RFLOW = "rflow"

class DPS_DDIMScheduler(Scheduler):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance. Based on: Song et al. "Denoising Diffusion
    Implicit Models" https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps: number of diffusion steps used to train the model.
        schedule: member of NoiseSchedules, name of noise schedule function in component store
        clip_sample: option to clip predicted sample between -1 and 1 for numerical stability.
        set_alpha_to_one: each diffusion step uses the value of alphas product at that step and at the previous one.
            For the final step there is no previous alpha. When this option is `True` the previous alpha product is
            fixed to `1`, otherwise it uses the value of alpha at step 0.
            A similar approach is used for reverse steps, setting this option to `True` will use zero as
            the first alpha.
        steps_offset: an offset added to the inference steps. You can use a combination of `steps_offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
        prediction_type: member of DDPMPredictionType
        schedule_args: arguments to pass to the schedule function

    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = DDIMPredictionType.EPSILON,
        clip_sample_min: float = -1.0,
        clip_sample_max: float = 1.0,
        **schedule_args,
    ) -> None:
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if prediction_type not in DDIMPredictionType.__members__.values():
            raise ValueError("Argument `prediction_type` must be a member of DDIMPredictionType")

        self.prediction_type = prediction_type

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # For reverse steps, we require the next alphas_cumprod. Similary to above, the first step doesn't
        # have a next value so we can either set it to zero or use the second value.
        self.first_alpha_cumprod = (
            torch.tensor(0.0) if set_alpha_to_one else self.alphas_cumprod[-1]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].astype(np.int64)
        )

        self.clip_sample = clip_sample
        self.clip_sample_min = clip_sample_min
        self.clip_sample_max = clip_sample_max
        self.steps_offset = steps_offset

        # default the number of inference timesteps to the number of train steps
        self.set_timesteps(self.num_train_timesteps)

    def set_timesteps(
        self, num_inference_steps: int, device: str | torch.device | None = None
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
            device: target device to put the data.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        # Full-range linspace spacing so the first (highest) timestep is always
        # T-1, regardless of num_inference_steps. The old `step_ratio = T // N`
        # truncated the top of the schedule when N did not divide T (at the
        # solver's 32 steps: highest timestep 961 instead of 999, so DDIM-prior
        # inverse problems began from pure noise treated as already partly
        # denoised). Mirrors the same fix in ddpm3d_mri/scheduler.py:DDIMScheduler.
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1].copy().astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset
        
    def step(
        self,
        model,
        forward_class,
        y,
        t,
        x_t,
        zeta: float = 1.0,
        sigma: float | None = None,
        next_timestep=None,
        eta=None,
        noise_model: str = "gaussian",
        data_consistency_steps: int = 50,
        loss_reduction: str = "batch_sum",
    ):

        noise_model = noise_model.lower()
    
        # Make x_t differentiable for DPS guidance.
        # The model output is computed without gradients, as usual for DPS.
        x_t = x_t.detach().requires_grad_(True)
    
        with torch.no_grad(), autocast(device_type="cuda", enabled=x_t.is_cuda):
            model_output = model(x_t, timesteps=torch.Tensor((t,)).to(x_t.device))
    
        # Previous timestep from the ACTUAL linspace schedule, not the integer
        # `t - T // N` offset (which under-steps the full-range schedule; see
        # scheduler.scheduled_neighbor / ADR-0003). At 32 steps the error is mild,
        # but this keeps the inverse solver consistent with the sampler.
        s = scheduled_neighbor(
            self.timesteps, t, self.num_train_timesteps, self.num_inference_steps,
        )

        α_bar_t = self.alphas_cumprod[t]
        β_bar_t = 1 - α_bar_t

        α_bar_s = self.alphas_cumprod[s] if s >= 0 else self.final_alpha_cumprod
        β_bar_s = 1 - α_bar_s

        α_sqrt = α_bar_t**0.5
        β_sqrt = β_bar_t**0.5

        # Get clean and noisy predictions from time t
        # Similar logic as `decompose_prediction`... maybe we can
        # use that and do some decoupling in the future...
        match self.prediction_type:
            case DDIMPredictionType.EPSILON:
                x_0_given_t = (x_t - β_sqrt * model_output) / α_sqrt
            case DDIMPredictionType.SAMPLE:
                x_0_given_t = model_output + (x_t - x_t.detach()) # straight-through estimation
            case DDIMPredictionType.VELOCITY:
                x_0_given_t = α_sqrt * x_t - β_sqrt * model_output
            case DDIMPredictionType.FLOW:
                x_0_given_t = (x_t - β_sqrt * model_output) / (α_sqrt + β_sqrt)

        # Clip for stability
        x_0_given_t = torch.clamp(x_0_given_t, self.clip_sample_min, self.clip_sample_max)

        # Recompute epsilon from the CLIPPED x_0 so (x_0, epsilon) stay a consistent
        # decomposition of x_t (mu_s below uses both). Algebraically identical to the
        # per-prediction-type formula when the clamp is inactive; only the early,
        # clamped high-noise steps change. Mirrors the sampler (scheduler.step 3b).
        ε_hat_t = (x_t - α_sqrt * x_0_given_t) / β_sqrt

        # ===== DPS likelihood guidance =====
        recon = forward_class(x_0_given_t)

        loss = measurement_loss(
            recon=recon,
            y=y,
            noise_model=noise_model,
            sigma=sigma,
            reduction=loss_reduction,
        )

        grad = torch.autograd.grad(outputs=loss, inputs=x_t)[0]
    
        # ===== Resume DDIM as normal =====
        σ2_t = (β_bar_s / β_bar_t) * (1 - α_bar_t / α_bar_s)
        σ_t = eta * σ2_t**0.5
        z = torch.randn_like(model_output)

        # Clean state at time s
        μ_s = α_bar_s**0.5 * x_0_given_t + (1 - α_bar_s - σ2_t)**0.5 * ε_hat_t

        # DPS
        μ_s -= zeta * grad
        
        # Sampled state at time s
        x_s = μ_s + σ_t * z
    
        return x_s.detach()



class DPS_RFlowScheduler(Scheduler):
    """
    Diffusion Posterior Sampling modification of the RFlow Scheduler.
    
    A rectified flow scheduler for guiding the diffusion process in a generative model.

    Supports uniform and logit-normal sampling methods and noise addition during diffusion.

    Args:
        num_train_timesteps (int): Total number of training timesteps.
        use_discrete_timesteps (bool): Whether to use discrete timesteps.
        sample_method (str): Training time step sampling method ('uniform' or 'logit-normal').
        loc (float): Location parameter for logit-normal distribution, used only if sample_method='logit-normal'.
        scale (float): Scale parameter for logit-normal distribution, used only if sample_method='logit-normal'.

    Example:

        .. code-block:: python

            # define a scheduler
            noise_scheduler = RFlowScheduler(
                num_train_timesteps = 1000,
                use_discrete_timesteps = True,
                sample_method = 'logit-normal',
            )

            # during training
            inputs = torch.ones(2,4,64,64,32)
            noise = torch.randn_like(inputs)
            timesteps = noise_scheduler.sample_timesteps(inputs)
            noisy_inputs = noise_scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            predicted_velocity = diffusion_unet(
                x=noisy_inputs,
                timesteps=timesteps
            )
            loss = loss_l1(predicted_velocity, (inputs - noise))

            # during inference
            inferer = DiffusionInferer(noise_scheduler)
            noisy_inputs = torch.randn(2,4,64,64,32)
            noise_scheduler.set_timesteps(num_inference_steps=30)
            final_output = inferer.sample(
                input_noise=noisy_inputs,
                diffusion_model=diffusion_unet,
           )
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        use_discrete_timesteps: bool = True,
        sample_method: str = "uniform",
        loc: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__(num_train_timesteps)
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        if sample_method not in ["uniform", "logit-normal"]:
            raise ValueError(
                f"sample_method = {sample_method}, which has to be chosen from ['uniform', 'logit-normal']."
            )
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)
        elif sample_method == "uniform":
            self.sample_t = lambda x: torch.rand((x.shape[0],), device=x.device)
        else:
            raise ValueError(f"`sample_method` has to be one of ['uniform', 'logit-normal'].")

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the original samples.

        Args:
            original_samples: original samples
            noise: noise to add to samples
            timesteps: timesteps tensor with shape of (N,), indicating the timestep to be computed for each sample.

        Returns:
            noisy_samples: sample with added noise
        """
        timepoints: torch.Tensor = timesteps.float() / self.num_train_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # expand timepoint to noise shape
        if noise.ndim == 5:
            timepoints = timepoints[..., None, None, None, None].expand(-1, *noise.shape[1:])
        elif noise.ndim == 4:
            timepoints = timepoints[..., None, None, None].expand(-1, *noise.shape[1:])
        else:
            raise ValueError(
                f"noise tensor has to be 4D or 5D tensor, yet got shape of {noise.shape}"
            )

        noisy_samples: torch.Tensor = timepoints * original_samples + (1 - timepoints) * noise

        return noisy_samples

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
            device: target device to put the data.
        """
        if num_inference_steps > self.num_train_timesteps or num_inference_steps < 1:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} should be at least 1, "
                "and cannot be larger than `self.num_train_timesteps`"
            )

        self.num_inference_steps = num_inference_steps
        # prepare timesteps
        timesteps = torch.as_tensor([
            (1.0 - i / self.num_inference_steps) * self.num_train_timesteps
            for i in range(self.num_inference_steps)
        ]).float()
        if self.use_discrete_timesteps:
            timesteps = timesteps.round().long()
        self.timesteps = timesteps.to(device)

    def sample_timesteps(self, x_start):
        """
        Randomly samples training timesteps using the chosen sampling method.

        Args:
            x_start (torch.Tensor): The input tensor for sampling.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        t = self.sample_t(x_start) * self.num_train_timesteps

        if self.use_discrete_timesteps:
            t = t.round().long()

        return t

    # # FlowDPS
    # def step(
    #     self,
    #     model,
    #     forward_class,
    #     y,
    #     t,
    #     x_t,
    #     zeta=1.0,
    #     sigma=1.0,
    #     next_timestep=None,
    #     eta=None, # Unused argrument here just for consistency
    # ):
    #     """
    #     Variable notation
    #     s is the timestep closer to data along our schedule
    #     s < t

    #     x_s is the "state closer to data"
    #     x_t is the "state closer to noise" or the "current state"
    #     This is also the input the neural network.

    #     x_0_given_t is the clean prediction from the state x_t

    #     v_hat_t is the estimated rectified-flow velocity at time t

    #     In rectified flow, the forward noising convention used by RFlowScheduler is

    #         x_t = (1 - t / T) * x_0 + (t / T) * noise

    #     and the model is trained to predict

    #         v_hat_t = x_0 - noise

    #     Therefore,

    #         x_0_given_t = x_t + (t / T) * v_hat_t

    #     and one deterministic reverse step from t to s is

    #         x_s = x_t + ((t - s) / T) * v_hat_t
    #     """
    #     # Ensure num_inference_steps exists and is a valid integer
    #     if not hasattr(self, "num_inference_steps") or not isinstance(
    #         self.num_inference_steps, int
    #     ):
    #         raise AttributeError(
    #             "num_inference_steps is missing or not an integer in the class. "
    #             "Please run self.set_timesteps(num_inference_steps, device) to set it."
    #         )
            
    #     x_t = x_t.detach()

    #     with torch.no_grad(), autocast(device_type="cuda", enabled=True):
    #         v_hat = model(x_t, timesteps=torch.Tensor((t,)).to(x_t.device))

    #     if next_timestep is not None:
    #         s = next_timestep
    #     else:
    #         s = t - self.num_train_timesteps / self.num_inference_steps

    #     if self.use_discrete_timesteps:
    #         s = int(round(float(s)))

    #     # Normalized RF time increments.
    #     t_over_T = float(t) / float(self.num_train_timesteps)
    #     s_over_T = float(s) / float(self.num_train_timesteps)
    #     dt = float(t - s) / float(self.num_train_timesteps)

    #     # Clean prediction from current state x_t.
    #     x0_hat = x_t + t_over_T * v_hat
    #     noise_hat = x_t - (1.0 - t_over_T) * v_hat

    #     with autocast(device_type="cuda", enabled=True):                        
    #         x0_corr = x0_hat
    #         for _ in range(50): # data consistency iterations
    #             x0_corr = x0_corr.detach().requires_grad_(True)
    #             recon = forward_class(x0_corr)
    #             loss = torch.linalg.norm((recon - y).reshape(x0_corr.shape[0], -1), dim=1).sum()
    #             grad = torch.autograd.grad(loss, x0_corr)[0]
    #             x0_corr = x0_corr - zeta * grad
            
    #     # FlowDPS-style time-dependent blending
    #     x0_corr = (1.0 - t_over_T) * x0_hat + t_over_T * x0_corr

    #     # Official FlowDPS-style stochastic renoising of the noise component.        
    #     if s_over_T > 0:
    #         noise_hat = math.sqrt(s_over_T) * noise_hat + math.sqrt(1.0 - s_over_T) * torch.randn_like(noise_hat)

    #     x_s = (1.0 - s_over_T) * x0_corr + s_over_T * noise_hat
    #     return x_s.detach()


    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timepoints: torch.Tensor = timesteps.float() / self.num_train_timesteps
        timepoints = 1 - timepoints

        if noise.ndim == 5:
            timepoints = timepoints[..., None, None, None, None].expand(-1, *noise.shape[1:])
        elif noise.ndim == 4:
            timepoints = timepoints[..., None, None, None].expand(-1, *noise.shape[1:])
        else:
            raise ValueError(
                f"noise tensor has to be 4D or 5D tensor, yet got shape of {noise.shape}"
            )

        noisy_samples: torch.Tensor = timepoints * original_samples + (1 - timepoints) * noise
        return noisy_samples

    def step(
        self,
        model,
        forward_class,
        y,
        t,
        x_t,
        zeta: float = 1.0,
        sigma: float | None = None,
        next_timestep=None,
        eta=None,
        noise_model: str = "gaussian",
        data_consistency_steps: int = 50,
        loss_reduction: str = "batch_sum",
        stochastic_renoising: bool = True,
    ):
        noise_model = noise_model.lower()
        if noise_model in {"rician", "rice"} and sigma is None:
            raise ValueError("sigma must be provided when noise_model='rician'.")

        x_t = x_t.detach()

        # Model velocity.
        timestep_tensor = _expand_batch_timesteps(t, x_t)

        with torch.no_grad(), autocast(device_type="cuda", enabled=x_t.is_cuda):
            v_hat = model(x_t, timesteps=timestep_tensor)

        if next_timestep is not None:
            s = next_timestep
        else:
            s = t - self.num_train_timesteps / self.num_inference_steps

        if self.use_discrete_timesteps:
            s = int(round(float(s)))

        t_over_T = float(t) / float(self.num_train_timesteps)
        s_over_T = float(s) / float(self.num_train_timesteps)

        # Rectified-flow Tweedie-style estimates:
        #
        #   x_t = (1 - t/T) x_0 + (t/T) noise
        #   model predicts v = x_0 - noise
        #
        # Therefore:
        #   x0_hat    = x_t + (t/T) v
        #   noise_hat = x_t - (1 - t/T) v
        x0_hat = x_t + t_over_T * v_hat
        noise_hat = x_t - (1.0 - t_over_T) * v_hat

        # FlowDPS likelihood update on the clean component.
        x0_corr = x0_hat.detach()

        with autocast(device_type="cuda", enabled=x_t.is_cuda):
            for _ in range(data_consistency_steps):
                x0_corr = x0_corr.detach().requires_grad_(True)

                recon = forward_class(x0_corr)

                loss = measurement_loss(
                    recon=recon,
                    y=y,
                    noise_model=noise_model,
                    sigma=sigma,
                    reduction=loss_reduction,
                )

                grad = torch.autograd.grad(loss, x0_corr)[0]
                x0_corr = x0_corr - zeta * grad

        # FlowDPS-style time-dependent blending.
        #
        # In the FlowDPS paper, the clean update is emphasized early and relaxed late.
        # Since t/T is large early and small late, this is equivalent to gamma_t = t/T.
        x0_corr = (1.0 - t_over_T) * x0_hat + t_over_T * x0_corr

        # FlowDPS-style stochastic renoising of the noise component.
        if stochastic_renoising and s_over_T > 0:
            noise_hat = (
                math.sqrt(s_over_T) * noise_hat
                + math.sqrt(1.0 - s_over_T) * torch.randn_like(noise_hat)
            )

        x_s = (1.0 - s_over_T) * x0_corr + s_over_T * noise_hat
        return x_s.detach()

    

        # This is just normal DPS, but it doesn't work in general for rect flow
        # def step(
        #     self,
        #     model,
        #     forward_class,
        #     y,
        #     t,
        #     x_t,
        #     zeta=1.0,
        #     sigma=1.0,
        #     next_timestep=None,
        #     eta=None, # Unused argrument here just for consistency
        # ):
        #     """
        #     Variable notation
        #     s is the timestep closer to data along our schedule
        #     s < t
    
        #     x_s is the "state closer to data"
        #     x_t is the "state closer to noise" or the "current state"
        #     This is also the input the neural network.
    
        #     x_0_given_t is the clean prediction from the state x_t
    
        #     v_hat_t is the estimated rectified-flow velocity at time t
    
        #     In rectified flow, the forward noising convention used by RFlowScheduler is
    
        #         x_t = (1 - t / T) * x_0 + (t / T) * noise
    
        #     and the model is trained to predict
    
        #         v_hat_t = x_0 - noise
    
        #     Therefore,
    
        #         x_0_given_t = x_t + (t / T) * v_hat_t
    
        #     and one deterministic reverse step from t to s is
    
        #         x_s = x_t + ((t - s) / T) * v_hat_t
        #     """
    
        #     # Ensure num_inference_steps exists and is a valid integer
        #     if not hasattr(self, "num_inference_steps") or not isinstance(
        #         self.num_inference_steps, int
        #     ):
        #         raise AttributeError(
        #             "num_inference_steps is missing or not an integer in the class. "
        #             "Please run self.set_timesteps(num_inference_steps, device) to set it."
        #         )
    
        #     # We need gradients through x_t for DPS, but not through the model weights.
        #     # This mirrors DPS_DDIMScheduler: model execution is no-grad, while the
        #     # measurement loss remains differentiable with respect to x_t.
        #     x_t = x_t.detach().requires_grad_(True)
    
        #     with torch.no_grad(), autocast(device_type="cuda", enabled=True):
        #         model_output = model(x_t, timesteps=torch.Tensor((t,)).to(x_t.device))
    
        #     # In rectified flow, the model output is the velocity prediction.
        #     v_hat_t = model_output
    
        #     if next_timestep is not None:
        #         s = next_timestep
        #     else:
        #         s = t - self.num_train_timesteps / self.num_inference_steps
    
        #     if self.use_discrete_timesteps:
        #         s = int(round(float(s)))
    
        #     # Normalized RF time increments.
        #     t_over_T = float(t) / float(self.num_train_timesteps)
        #     dt = float(t - s) / float(self.num_train_timesteps)
    
        #     # Clean prediction from current state x_t.
        #     x_0_given_t = x_t + t_over_T * v_hat_t
    
        #     # ===== DPS custom config =====
    
        #     recon = forward_class(x_0_given_t)
        #     err = recon - y
        #     norm = err.pow(2).sum()
        #     grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
    
        #     # ===== Resume RFlow as normal =====
        #     # Deterministic rectified-flow state at time s.
        #     μ_s = x_t + dt * v_hat_t
    
        #     # DPS
        #     μ_s -= zeta * grad
    
        #     # Sampled state at time s.
        #     # Rectified flow is deterministic here, so there is no DDIM-style eta noise term.
        #     x_s = μ_s
    
        #     return x_s.detach()



# -------------------------------------------------------------------------
# Default measurement loss
# -------------------------------------------------------------------------
# If your project already has measurement_loss, delete this or rename it.
# This version is intentionally simple.

def default_measurement_loss(
    recon: torch.Tensor,
    y: torch.Tensor,
    noise_model: str = "gaussian",
    sigma: Optional[float] = None,
    reduction: Literal["mean", "sum", "batch_sum"] = "batch_sum",
) -> torch.Tensor:
    noise_model = noise_model.lower()

    if noise_model in {"gaussian", "normal", "l2"}:
        resid = recon - y
        loss_per_elem = resid.square()

        if sigma is not None:
            loss_per_elem = loss_per_elem / (2.0 * float(sigma) ** 2)

    elif noise_model in {"rician", "rice"}:
        # Placeholder. Prefer your existing Rician loss if you have one.
        if sigma is None:
            raise ValueError("sigma must be provided for Rician noise.")
        resid = recon - y
        loss_per_elem = resid.square() / (2.0 * float(sigma) ** 2)

    else:
        raise ValueError(f"Unknown noise_model={noise_model!r}")

    if reduction == "mean":
        return loss_per_elem.mean()
    if reduction == "sum":
        return loss_per_elem.sum()
    if reduction == "batch_sum":
        return loss_per_elem.flatten(start_dim=1).sum(dim=1).mean()

    raise ValueError(f"Unknown reduction={reduction!r}")


# -------------------------------------------------------------------------
# RF stochastic resampling
# -------------------------------------------------------------------------

def default_resample_sigma2_rf(
    tau: float,
    gamma: float = 40.0,
    eps: float = 1e-8,
) -> float:
    """
    Rectified-flow analogue of ReSample's stochastic-resampling variance.

    RF forward convention:
        z_tau = (1 - tau) z0 + tau eps

    Prior variance is tau^2, so we use:
        sigma_t^2 = gamma * tau^2
    """
    q = max(float(tau) ** 2, eps)
    return float(gamma) * q


def stochastic_resample_rf(
    z0_y: torch.Tensor,
    z_s_prime: torch.Tensor,
    tau_s: float,
    sigma2: float,
    add_noise: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    RF analogue of stochastic resampling.

    Treat:
        z_s | z0_y ~ N((1 - tau_s) z0_y, tau_s^2 I)

    and combine it with the unconditional sample z_s_prime via Gaussian product logic.
    """
    tau_s = float(tau_s)
    q = max(tau_s ** 2, eps)

    sigma2 = float(sigma2)
    denom = sigma2 + q

    mean = (sigma2 * (1.0 - tau_s) * z0_y + q * z_s_prime) / denom
    var = (sigma2 * q) / denom

    if add_noise and var > 0:
        return mean + math.sqrt(var) * torch.randn_like(mean)

    return mean


# -------------------------------------------------------------------------
# Pixel-space hard data consistency: differentiable forward model version
# -------------------------------------------------------------------------

def hard_data_consistency_pixel_gd(
    x_init: torch.Tensor,
    forward_class: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    measurement_loss_fn: Callable = default_measurement_loss,
    noise_model: str = "gaussian",
    sigma: Optional[float] = None,
    lr: float = 1e-2,
    max_iters: int = 100,
    tau_stop: Optional[float] = 1e-4,
    loss_reduction: str = "batch_sum",
    optimizer_name: Literal["adam", "adamw", "sgd"] = "adamw",
    clamp: bool = False,
    xmin: float = -1.0,
    xmax: float = 1.0,
    grad_clip: Optional[float] = None,
    use_amp: bool = True,
    empty_cache_each_iter: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Pixel/image-space hard DC:

        min_x  measurement_loss(A(x), y)

    initialized at x_init = D(z0_hat).

    This does NOT backpropagate through the decoder. It only optimizes x.
    """
    # Run GD in fp32. The nonlinear operators here use cuFFT (kspace/motion),
    # which rejects non-power-of-2 signal sizes in half precision, and the
    # batch_sum loss over millions of voxels overflows fp16 to nan. x_init
    # arrives from the decoder in fp16, so upcast it (and y). autocast is forced
    # off below for the same reason, so use_amp is intentionally ignored here.
    x = x_init.detach().clone().float().requires_grad_(True)
    y = y.float()

    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        opt = torch.optim.Adam([x], lr=lr)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW([x], lr=lr)
    elif opt_name == "sgd":
        opt = torch.optim.SGD([x], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name!r}")

    last_loss = None

    for k in range(max_iters):
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=False):
            recon = forward_class(x)
            loss = measurement_loss_fn(
                recon=recon,
                y=y,
                noise_model=noise_model,
                sigma=sigma,
                reduction=loss_reduction,
            )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([x], grad_clip)

        opt.step()

        if clamp:
            with torch.no_grad():
                x.clamp_(xmin, xmax)

        last_loss = float(loss.detach().cpu())

        del recon, loss
        _maybe_empty_cache(empty_cache_each_iter)

        if tau_stop is not None and last_loss <= tau_stop:
            break

    if verbose:
        print(f"[pixel GD DC] iters={k + 1}, final_loss={last_loss:.6e}")

    return x.detach()


# -------------------------------------------------------------------------
# Pixel-space hard data consistency: CG proximal version for linear A
# -------------------------------------------------------------------------

def _dot_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batchwise inner product, returned as shape [B].
    """
    return (x.flatten(start_dim=1) * y.flatten(start_dim=1)).sum(dim=1)


@torch.no_grad()
def conjugate_gradient_batch(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iters: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Batched conjugate gradient for solving:

        matvec(x) = b

    where x and b are image-shaped tensors [B, C, ...].
    """
    x = torch.zeros_like(b) if x0 is None else x0.detach().clone()

    r = b - matvec(x)
    p = r.clone()
    rsold = _dot_batch(r, r)

    for k in range(max_iters):
        Ap = matvec(p)
        denom = _dot_batch(p, Ap).clamp_min(eps)
        alpha = rsold / denom

        view_shape = [alpha.shape[0]] + [1] * (x.ndim - 1)
        alpha_v = alpha.view(*view_shape)

        x = x + alpha_v * p
        r = r - alpha_v * Ap

        rsnew = _dot_batch(r, r)

        if torch.sqrt(rsnew.max()).item() < tol:
            rsold = rsnew
            break

        beta = (rsnew / rsold.clamp_min(eps)).view(*view_shape)
        p = r + beta * p
        rsold = rsnew

    if verbose:
        print(f"[CG] iters={k + 1}, max_resid={torch.sqrt(rsold.max()).item():.6e}")

    return x.detach()


@torch.no_grad()
def hard_data_consistency_pixel_cg(
    x_init: torch.Tensor,
    y: torch.Tensor,
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    AT_fn: Callable[[torch.Tensor], torch.Tensor],
    lam: float = 1e-2,
    max_iters: int = 20,
    tol: float = 1e-6,
    clamp: bool = False,
    xmin: float = -1.0,
    xmax: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Linear proximal data consistency via CG.

    Solves:

        x_y = argmin_x 0.5 ||A x - y||_2^2 + 0.5 * lam ||x - x_init||_2^2

    Normal equations:

        (A^T A + lam I) x = A^T y + lam x_init

    This uses no autograd and is usually much more memory-friendly than GD.
    """
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be positive for stable proximal CG data consistency.")

    def matvec(x):
        return AT_fn(A_fn(x)) + lam * x

    b = AT_fn(y) + lam * x_init

    x = conjugate_gradient_batch(
        matvec=matvec,
        b=b,
        x0=x_init,
        max_iters=max_iters,
        tol=tol,
        verbose=verbose,
    )

    if clamp:
        x = x.clamp(xmin, xmax)

    return x.detach()


def hard_data_consistency_pixel_encode_back(
    z0_init: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    dc_mode: Literal["gd", "cg"] = "gd",

    # GD mode args
    forward_class: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    measurement_loss_fn: Callable = default_measurement_loss,
    noise_model: str = "gaussian",
    sigma: Optional[float] = None,
    gd_lr: float = 1e-2,
    gd_max_iters: int = 100,
    gd_tau_stop: Optional[float] = 1e-4,
    gd_loss_reduction: str = "batch_sum",
    gd_optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
    gd_grad_clip: Optional[float] = None,

    # CG mode args
    A_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    AT_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cg_lam: float = 1e-2,
    cg_max_iters: int = 20,
    cg_tol: float = 1e-6,

    # Shared args
    clamp_x: bool = False,
    xmin: float = -1.0,
    xmax: float = 1.0,
    use_amp_decode_encode: bool = True,
    use_amp_gd: bool = True,
    empty_cache: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Official-implementation-style hard DC surrogate:

        z0_init
          -> x_init = D(z0_init)          [no grad]
          -> x_y = pixel hard DC          [GD or CG]
          -> z_y = E(x_y)                 [no grad]
          -> z0_y = blend*z_y + (1-blend)*z0_init

    This avoids decoder-backprop entirely.
    """
    # Decode without grad.
    with torch.no_grad(), autocast(device_type="cuda", enabled=(use_amp_decode_encode and z0_init.is_cuda)):
        x_init = decode_fn(z0_init.detach())
        if clamp_x:
            x_init = x_init.clamp(xmin, xmax)

    _maybe_empty_cache(empty_cache)

    dc_mode = dc_mode.lower()

    if dc_mode == "gd":
        if forward_class is None:
            raise ValueError("forward_class must be provided when dc_mode='gd'.")

        x_y = hard_data_consistency_pixel_gd(
            x_init=x_init,
            forward_class=forward_class,
            y=y,
            measurement_loss_fn=measurement_loss_fn,
            noise_model=noise_model,
            sigma=sigma,
            lr=gd_lr,
            max_iters=gd_max_iters,
            tau_stop=gd_tau_stop,
            loss_reduction=gd_loss_reduction,
            optimizer_name=gd_optimizer,
            clamp=clamp_x,
            xmin=xmin,
            xmax=xmax,
            grad_clip=gd_grad_clip,
            use_amp=use_amp_gd,
            empty_cache_each_iter=empty_cache,
            verbose=verbose,
        )

    elif dc_mode == "cg":
        if A_fn is None or AT_fn is None:
            raise ValueError("A_fn and AT_fn must be provided when dc_mode='cg'.")

        # Run CG in fp32, NOT under autocast. The dot-products sum over millions
        # of voxels and overflow fp16 to inf, while the fp32-scale eps guard
        # underflows to 0 in fp16 -> inf/0 -> nan residuals. x_init comes back
        # from the decoder in fp16, so cast it (and y) up explicitly.
        x_y = hard_data_consistency_pixel_cg(
            x_init=x_init.float(),
            y=y.float(),
            A_fn=A_fn,
            AT_fn=AT_fn,
            lam=cg_lam,
            max_iters=cg_max_iters,
            tol=cg_tol,
            clamp=clamp_x,
            xmin=xmin,
            xmax=xmax,
            verbose=verbose,
        )

    else:
        raise ValueError("dc_mode must be 'gd' or 'cg'.")

    del x_init
    _maybe_empty_cache(empty_cache)

    # Encode without grad.
    with torch.no_grad(), autocast(device_type="cuda", enabled=(use_amp_decode_encode and x_y.is_cuda)):
        z_y = encode_fn(x_y.detach())

    del x_y
    _maybe_empty_cache(empty_cache)

    return z_y.detach()


class ReSampleRFlowPixelDCScheduler(DPS_RFlowScheduler):
    """
    Rectified-flow ReSample with pixel-space hard data consistency + encode-back.

    Assumes RF convention:
        z_tau = (1 - tau) z0 + tau eps
        model predicts v = z0 - eps

    Reverse step:
        1. Predict z0_hat from z_t.
        2. Take unconditional RF step z_s_prime.
        3. At selected steps:
             z0_hat -> decode no_grad -> pixel DC -> encode no_grad -> z0_y
        4. Stochastically resample z0_y back to time s.

    This avoids differentiating through the decoder.
    """

    def should_resample(
        self,
        step_index: int,
        num_steps: int,
        resample_every: int = 10,
        start_fraction: float = 1.0 / 3.0,
    ) -> bool:
        if resample_every <= 0:
            return False

        if step_index < int(start_fraction * num_steps):
            return False

        return (step_index % resample_every) == 0 or (step_index == num_steps - 1)

    def step(
        self,
        model,
        forward_class,
        y,
        t,
        z_t,
        decode_fn,
        encode_fn,
        step_index: int,

        # Optional explicit next timestep
        next_timestep=None,

        # ReSample controls
        use_resample: bool = True,
        resample_every: int = 10,
        resample_start_fraction: float = 1.0 / 3.0,
        resample_gamma: float = 40.0,
        resample_add_noise: bool = True,

        # Pixel hard DC mode
        dc_mode: Literal["gd", "cg"] = "gd",

        # GD hard DC args
        measurement_loss_fn: Callable = default_measurement_loss,
        noise_model: str = "gaussian",
        sigma: Optional[float] = None,
        gd_lr: float = 1e-2,
        gd_max_iters: int = 100,
        gd_tau_stop: Optional[float] = 1e-4,
        gd_loss_reduction: str = "batch_sum",
        gd_optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
        gd_grad_clip: Optional[float] = None,

        # CG hard DC args
        A_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        AT_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cg_lam: float = 1e-2,
        cg_max_iters: int = 20,
        cg_tol: float = 1e-6,

        # Numerical/image controls
        clamp_x: bool = False,
        xmin: float = -1.0,
        xmax: float = 1.0,
        use_amp_model: bool = True,
        use_amp_decode_encode: bool = True,
        use_amp_gd: bool = True,
        empty_cache: bool = True,
        verbose_dc: bool = False,

        # Keep default off unless separately validated
        stochastic_renoising: bool = False,
    ):
        noise_model = noise_model.lower()
        if noise_model in {"rician", "rice"} and sigma is None and dc_mode == "gd":
            raise ValueError("sigma must be provided when noise_model='rician'.")

        z_t = z_t.detach()

        timestep_tensor = _expand_batch_timesteps(t, z_t)

        # Unconditional RF model prediction. No sampler graph retained.
        with torch.inference_mode(), autocast(device_type="cuda", enabled=(use_amp_model and z_t.is_cuda)):
            v_hat = model(z_t, timesteps=timestep_tensor)

        if next_timestep is not None:
            s = next_timestep
        else:
            s = float(t) - self.num_train_timesteps / self.num_inference_steps

        if self.use_discrete_timesteps:
            s = int(round(float(s)))

        tau_t = float(t) / float(self.num_train_timesteps)
        tau_s = float(s) / float(self.num_train_timesteps)

        tau_t = min(max(tau_t, 0.0), 1.0)
        tau_s = min(max(tau_s, 0.0), 1.0)

        # RF Tweedie-style estimates:
        # z_t = (1 - tau_t) z0 + tau_t eps
        # v = z0 - eps
        z0_hat = z_t + tau_t * v_hat
        noise_hat = z_t - (1.0 - tau_t) * v_hat

        if stochastic_renoising and tau_s > 0:
            noise_hat = (
                math.sqrt(tau_s) * noise_hat
                + math.sqrt(max(1.0 - tau_s, 0.0)) * torch.randn_like(noise_hat)
            )

        # Unconditional step to s.
        z_s_prime = (1.0 - tau_s) * z0_hat + tau_s * noise_hat

        # Detach everything before hard DC.
        z0_init = z0_hat.detach()
        z_s_prime = z_s_prime.detach()

        del z_t, v_hat, noise_hat, z0_hat
        _maybe_empty_cache(empty_cache)

        do_resample = (
            use_resample
            and self.should_resample(
                step_index=step_index,
                num_steps=self.num_inference_steps,
                resample_every=resample_every,
                start_fraction=resample_start_fraction,
            )
        )

        if do_resample:
            z0_y = hard_data_consistency_pixel_encode_back(
                z0_init=z0_init,
                decode_fn=decode_fn,
                encode_fn=encode_fn,
                y=y,
                dc_mode=dc_mode,

                # GD mode
                forward_class=forward_class,
                measurement_loss_fn=measurement_loss_fn,
                noise_model=noise_model,
                sigma=sigma,
                gd_lr=gd_lr,
                gd_max_iters=gd_max_iters,
                gd_tau_stop=gd_tau_stop,
                gd_loss_reduction=gd_loss_reduction,
                gd_optimizer=gd_optimizer,
                gd_grad_clip=gd_grad_clip,

                # CG mode
                A_fn=A_fn,
                AT_fn=AT_fn,
                cg_lam=cg_lam,
                cg_max_iters=cg_max_iters,
                cg_tol=cg_tol,

                # Shared
                clamp_x=clamp_x,
                xmin=xmin,
                xmax=xmax,
                use_amp_decode_encode=use_amp_decode_encode,
                use_amp_gd=use_amp_gd,
                empty_cache=empty_cache,
                verbose=verbose_dc,
            )

            sigma2_resample = default_resample_sigma2_rf(
                tau=tau_s,
                gamma=resample_gamma,
            )

            z_s = stochastic_resample_rf(
                z0_y=z0_y,
                z_s_prime=z_s_prime,
                tau_s=tau_s,
                sigma2=sigma2_resample,
                add_noise=resample_add_noise and tau_s > 0,
            )

            del z0_y

        else:
            z_s = z_s_prime

        del z0_init, z_s_prime
        _maybe_empty_cache(empty_cache)

        return z_s.detach()