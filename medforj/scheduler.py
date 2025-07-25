import torch
import numpy as np
from monai.utils import StrEnum
from monai.networks.schedulers import Scheduler


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


class DDIMScheduler(Scheduler):
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
            raise ValueError(
                "Argument `prediction_type` must be a member of DDIMPredictionType"
            )

        self.prediction_type = prediction_type

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

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
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            eta: weight of noise for added noise in diffusion step.
            generator: random number generator.
            reverse: whether to take a reversed step.

        Returns:
            pred_step_sample: Predicted previous (or next) sample
            pred_original_sample: Predicted original sample
        """
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - model_output -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        # 2. get other timestep value (=t-1) or (t+1) if reverse=True
        # and calculate alpha_prod_t for other timestep
        if reverse:
            step_timestep = (
                timestep + self.num_train_timesteps // self.num_inference_steps
            )
            alpha_prod_t_step = (
                self.alphas_cumprod[step_timestep]
                if step_timestep < len(self.alphas_cumprod)
                else self.first_alpha_cumprod
            )
        else:
            step_timestep = (
                timestep - self.num_train_timesteps // self.num_inference_steps
            )
            alpha_prod_t_step = (
                self.alphas_cumprod[step_timestep]
                if step_timestep >= 0
                else self.final_alpha_cumprod
            )
        beta_prod_t_step = 1 - alpha_prod_t_step

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.prediction_type == DDIMPredictionType.EPSILON:
            pred_original_sample = (sample - (beta_prod_t**0.5) * model_output) / (
                alpha_prod_t**0.5
            )
            pred_epsilon = model_output
        elif self.prediction_type == DDIMPredictionType.SAMPLE:
            pred_original_sample = model_output
            pred_epsilon = (sample - (alpha_prod_t**0.5) * pred_original_sample) / (
                beta_prod_t**0.5
            )
        elif self.prediction_type == DDIMPredictionType.VELOCITY:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        elif self.prediction_type == DDIMPredictionType.FLOW:
            pred_original_sample = (sample - (beta_prod_t**0.5) * model_output) / (
                (alpha_prod_t**0.5) + (beta_prod_t**0.5)
            )
            pred_epsilon = (sample + (alpha_prod_t**0.5) * model_output) / (
                (alpha_prod_t**0.5) + (beta_prod_t**0.5)
            )
        else:
            raise ValueError("Invalid prediction type")

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, self.clip_sample_min, self.clip_sample_max
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t+−1)/(1 − α_t)) * sqrt(1 − α_t/α_t+−1)
        variance = (beta_prod_t_step / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_step
        )
        std_dev_t = eta * variance**0.5  # type: ignore

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (
            1 - alpha_prod_t_step - std_dev_t**2
        ) ** 0.5 * pred_epsilon

        # 7. compute x_t+-1 without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_step_sample = (
            alpha_prod_t_step**0.5 * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(
                model_output.shape, dtype=model_output.dtype, generator=generator
            ).to(device)

            pred_step_sample = pred_step_sample + std_dev_t * noise

        return pred_step_sample, pred_original_sample