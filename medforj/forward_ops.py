"""
To evaluate the MedForj priors for inverse problems, 
we evaluate a suite of forward problems. These are:
- Super-resolution of 2D slice-selection
- Inpainting with masks
- Denoising extreme Rician noise
- Motion correction
- Sparse phase encode k-space reconstruction (k-space accel) 
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional, Sequence


# Some Fourier helper functions
def fft3c(x: torch.Tensor) -> torch.Tensor:
    dims = (-3, -2, -1)
    x = torch.fft.ifftshift(x, dim=dims)
    return torch.fft.fftshift(torch.fft.fftn(x, dim=dims, norm="ortho"), dim=dims)


def ifft3c(k: torch.Tensor) -> torch.Tensor:
    dims = (-3, -2, -1)
    k = torch.fft.ifftshift(k, dim=dims)
    return torch.fft.fftshift(torch.fft.ifftn(k, dim=dims, norm="ortho"), dim=dims)


# Adding rician noise doesn't need an A since it's a noise model
# ***This means that solving rician noise makes use of A = Identity***
def add_rician_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Rician-corrupt a magnitude image in [-1, 1]; sigma is in [0, 1] units."""
    x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    n_real = sigma * torch.randn_like(x01)
    n_imag = sigma * torch.randn_like(x01)
    y01 = torch.sqrt((x01 + n_real) ** 2 + n_imag**2 + 1e-12)
    return 2.0 * y01 - 1.0

def rician_nll(x_hat, y, sigma):
    """Rician NLL (up to a constant) for images in [-1, 1]; sigma in [0, 1] units."""
    x = ((x_hat + 1) / 2).clamp(0, 1)
    y = (y + 1) / 2                      # Rician samples are >= 0; can exceed 1
    z = x * y / sigma**2
    log_i0 = torch.log(torch.special.i0e(z)) + z
    return (x.square() / (2 * sigma**2) - log_i0).sum()


class ForwardModel(nn.Module):
    """
    The forward model is y = A(x) + n for some noise distribution n.
    We model Gaussian and Rician here, using the Rician NLL in that case.
    """
    def __init__(self, A, noise_model='gaussian', sigma: float = 0.0):
        super().__init__()
        self.A = A
        self.noise_model = noise_model
        self.sigma = sigma

    def forward(self, x):
        Ax = self.A(x)
        if self.noise_model == 'gaussian':
            return Ax if self.sigma == 0 else Ax + self.sigma * torch.randn_like(Ax)
        elif self.noise_model == 'rician':
            return add_rician_noise(Ax, self.sigma)

    def loss(self, x_hat, y):
        Ax_hat = self.A(x_hat)
        if self.noise_model == 'gaussian':
            return torch.linalg.norm(Ax_hat - y)  # DPS-style unsquared norm
        elif self.noise_model == 'rician':
            return rician_nll(Ax_hat, y, self.sigma)
    
class ForwardOperator(nn.Module, ABC):
    """
    Base class for inverse-problem forward operators.
    Tracks the adjoint too.
    """

    is_linear: bool = False
    has_adjoint: bool = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement an adjoint."
        )

class ComposedOperator(ForwardOperator):
    """
    Compose operators as A = A_k o ... o A_2 o A_1.
    Take care: order matters.

    Example:
        A = ComposedOperator(
            SliceSelectionSuperResolution1D(...),
            RicianNoiseObservation(...),
        )

        y = A(x)
    """

    def __init__(self, *operators: ForwardOperator):
        super().__init__()
        self.operators = nn.ModuleList(operators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.operators:
            x = op(x)
        return x
        
class Identity(ForwardOperator):
    """
    Identity measurement operator: y = x.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SliceSelection(ForwardOperator):
    """
    Models slice selection as a 1D blur-and-downsample
    along a chosen spatial axis. Results in T-skip-G
    resolution, for thickness T and gap G.

    This is implemented as a Toeplitz-like matrix by
    sampling a Gaussian slice profile at the correct locations.
    """

    is_linear = True
    has_adjoint = True

    def __init__(
        self,
        hr_shape: int,
        hr_spacing_mm: float,
        slice_thickness_mm: float,
        slice_separation_mm: float,
        offset_mm: float = 0.0,
        axis: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.hr_shape = hr_shape
        self.hr_spacing_mm = hr_spacing_mm
        self.slice_thickness_mm = slice_thickness_mm
        self.slice_separation_mm = slice_separation_mm
        self.offset_mm = offset_mm
        self.axis = axis

        A = self.get_obs_matrix(
            hr_shape=hr_shape,
            hr_spacing_mm=hr_spacing_mm,
            slice_thickness_mm=slice_thickness_mm,
            slice_separation_mm=slice_separation_mm,
            offset_mm=offset_mm,
            dtype=dtype,
            device=torch.device(device),
        )

        # Register as buffer so it moves with .to(device), but is not trainable.
        self.register_buffer("A", A)

    @staticmethod
    def gaussian_basis(
        t: torch.Tensor,
        s: float,
        dx: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return (
            torch.exp(-(t**2) / (2 * s**2 + eps))
            / (math.sqrt(2 * math.pi) * s + eps)
            * dx
        )

    @staticmethod
    def fwhm_to_std(gamma: float) -> float:
        return gamma / (2 * math.sqrt(2 * math.log(2)))

    @classmethod
    def get_obs_matrix(
        cls,
        hr_shape: int,
        hr_spacing_mm: float,
        slice_thickness_mm: float,
        slice_separation_mm: float,
        offset_mm: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        sigma_sq = (
            cls.fwhm_to_std(slice_thickness_mm) ** 2
            - cls.fwhm_to_std(hr_spacing_mm) ** 2
        )

        sigma = math.sqrt(sigma_sq)

        out_size = int(hr_shape * hr_spacing_mm / slice_separation_mm)

        j = torch.arange(hr_shape, dtype=dtype, device=device).view(1, -1)
        i = torch.arange(out_size, dtype=dtype, device=device).view(-1, 1)

        x_inp = (j - 0.5 * (hr_shape - 1.0)) * hr_spacing_mm
        x_out = (i - 0.5 * (out_size - 1.0)) * slice_separation_mm + offset_mm
        positions = x_out - x_inp

        return cls.gaussian_basis(positions, sigma, hr_spacing_mm)

    def apply_mat(self, A, x, axis=2):
        match axis:
            case 0: return torch.einsum('ij,...jnk->...ink', A, x)
            case 1: return torch.einsum('ij,...njk->...nik', A, x)
            case 2: return torch.einsum('ij,...knj->...kni', A, x)
            case _: raise ValueError
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_mat(self.A, x, axis=self.axis)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.apply_mat(self.A.T, y, axis=self.axis)

class Masking(ForwardOperator):
    """
    Elementwise masking for inpainting or outpainting.
    """

    is_linear = True
    has_adjoint = True

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # For a real-valued binary mask, the adjoint is itself.
        return self.mask * y

class KSpaceMasking3D(ForwardOperator):
    """
    A model of accelerated 3D Cartesian acquisition of a magnitude image in [-1, 1]:

        y = 2 * |F^-1 M F (x + 1)/2| - 1

    M is a variable-density random mask over the phase-encode plane (D, H)
    with a fully sampled center, broadcast along the frequency-encode axis W.
    Nonlinear (magnitude), so no adjoint.
    """

    def __init__(
        self,
        spatial_shape=(192, 224, 192),
        acceleration: float = 12.0,
        center_fraction: float = 0.08,
        power: float = 4.0,
    ):
        super().__init__()
        pe_mask = self.variable_density_mask(
            spatial_shape[:2], acceleration, center_fraction, power
        )
        self.register_buffer("mask", pe_mask[..., None])  # (D, H, 1)

    @staticmethod
    def variable_density_mask(shape, acceleration, center_fraction, power):
        h, w = shape
        # Unseeded CPU generator uses torch's fixed default seed.
        generator = torch.Generator()

        # Sampling probability decays with distance from the k-space center.
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
        )
        radius = torch.sqrt(gy**2 + gx**2)
        prob = (1.0 - radius / radius.max()).clamp_min(0.0) ** power

        # Fully sampled low-frequency center.
        ch, cw = round(h * center_fraction), round(w * center_fraction)
        h0, w0 = h // 2 - ch // 2, w // 2 - cw // 2
        center = torch.zeros(h, w, dtype=torch.bool)
        center[h0:h0 + ch, w0:w0 + cw] = True

        # Fill the rest of the budget by weighted sampling outside the center.
        mask = center.float()
        remaining = round(h * w / acceleration) - int(center.sum())
        weights = prob[~center] + 1e-8
        idx = torch.multinomial(
            weights / weights.sum(), remaining, replacement=False, generator=generator
        )
        outside = torch.nonzero(~center)
        mask[outside[idx, 0], outside[idx, 1]] = 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = fft3c((x + 1.0) / 2.0) * self.mask.to(x.device)
        return 2.0 * ifft3c(k).abs() - 1.0


class MRIRigidMotion3D(ForwardOperator):
    """
    A single rigid head movement partway through a 3D Cartesian acquisition
    (TorchIO RandomMotion-style) for a magnitude image in [-1, 1]:

        y = 2 * Re{ F^-1 [ M F(T(x01)) + (1 - M) F(x01) ] } - 1,   x01 = (x + 1) / 2

    T is a fixed rigid transform (rotation + translation). M selects the first
    `time` fraction of k-space lines along the slow phase-encode axis (D). Those lines are acquired
    before the move. The remaining lines, which contain the k-space center, come
    from the original pose. This matches TorchIO for time < 0.5.
    Nonlinear (resampling + real part), so no adjoint.
    """

    def __init__(
        self,
        spatial_shape=(192, 224, 192),
        degrees=(6.0, 6.0, 6.0),         # rotations about (x, y, z)
        translation_mm=(6.0, 6.0, 6.0),  # shifts along (z, y, x); 1 mm isotropic voxels
        time: float = 0.45,
    ):
        super().__init__()
        R = self.rotation_matrix(*degrees)
        t = torch.tensor(translation_mm)

        # Physical coordinates (mm), ordered (z, y, x), centered on the volume.
        axes = [torch.arange(n) - (n - 1) / 2 for n in spatial_shape]
        r = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (D, H, W, 3)

        # Pull-back sampling: moved(r) = x(R^T (r - t)).
        src = (r - t) @ R
        # Normalize to [-1, 1] (align_corners=True) and reorder to (x, y, z) for grid_sample.
        half = torch.tensor([(n - 1) / 2 for n in spatial_shape])
        self.register_buffer("grid", (src / half).flip(-1)[None])  # (1, D, H, W, 3)

        # Phase-encode lines along the slow axis (D) acquired before the motion event.
        D = spatial_shape[0]
        self.register_buffer("moved_lines", (torch.arange(D) < int(D * time))[:, None, None])

    @staticmethod
    def rotation_matrix(rx_deg, ry_deg, rz_deg):
        """R = Rz @ Ry @ Rx, acting on (z, y, x) coordinates."""
        cx, sx = math.cos(math.radians(rx_deg)), math.sin(math.radians(rx_deg))
        cy, sy = math.cos(math.radians(ry_deg)), math.sin(math.radians(ry_deg))
        cz, sz = math.cos(math.radians(rz_deg)), math.sin(math.radians(rz_deg))
        Rx = torch.tensor([[cx, -sx, 0.0], [sx, cx, 0.0], [0.0, 0.0, 1.0]])
        Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rz = torch.tensor([[1.0, 0.0, 0.0], [0.0, cz, -sz], [0.0, sz, cz]])
        return Rz @ Ry @ Rx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W) in [-1, 1]
        x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)

        # Pad with the image minimum (as TorchIO/SimpleITK does): shift, zero-pad, shift back.
        c = x01.amin(dim=(-3, -2, -1), keepdim=True)
        grid = self.grid.expand(x.shape[0], -1, -1, -1, -1)
        moved = F.grid_sample(x01 - c, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=True) + c

        k = torch.where(self.moved_lines, fft3c(moved), fft3c(x01))
        return 2.0 * ifft3c(k).abs() - 1.0