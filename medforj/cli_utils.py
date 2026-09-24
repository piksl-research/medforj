"""
Shared helpers for the MedForj demo scripts.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from safetensors.torch import load_file
from monai.networks.nets import DiffusionModelUNet
from monai.utils import set_determinism

import medforj
from medforj.maisi_setup import get_ae

IMG_SHAPE = (192, 224, 192)
MAISI_CONFIG_FPATH = Path(medforj.__file__).parent / "config_maisi.json"
# strategy -> weight fname
WEIGHT_FNAMES = {
    "noise":     "MedForj-weights-noise_ema.safetensors",
    "clean":     "MedForj-weights-clean_ema.safetensors",
    "velocity":  "MedForj-weights-velocity_ema.safetensors",
    "flow":      "MedForj-weights-flow_ema.safetensors",
    "rflow":     "MedForj-weights-rflow_ema.safetensors",
    "ldm_rflow": "MedForj-weights-ldm_rflow_ema.safetensors",
}


def common_parser():
    """Arguments shared by all demo scripts. Use as a parent parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out-fpath", type=Path, required=True)
    parser.add_argument("--weight-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, choices=list(WEIGHT_FNAMES), default="flow")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def parse_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        print("GPU index not provided or no GPU support currently available.")
        print("!!! Running on CPU !!!")
        device = torch.device("cpu")
    return device


def setup(args):
    """Seed, make the output directory, and return the device."""
    set_determinism(args.seed)
    args.out_fpath.parent.mkdir(parents=True, exist_ok=True)
    return parse_device(args.gpu_id)


def amp_ctx(device):
    """fp16 autocast on GPU, no-op on CPU."""
    if device.type == "cuda":
        return autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_prediction_type(strategy):
    return "rflow" if strategy == "ldm_rflow" else strategy


def build_unet(strategy, latent_channels=1):
    """The diffusion UNet for a strategy. Shared by training and inference."""
    if strategy == "ldm_rflow":
        return DiffusionModelUNet(
            spatial_dims=3,
            in_channels=latent_channels,
            out_channels=latent_channels,
            channels=(64, 128, 256, 512),
            norm_num_groups=8,
            attention_levels=(False, False, True, True),
            num_res_blocks=2,
            num_head_channels=32,
            use_flash_attention=True,
            cast_after_norm=True,
        )
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        norm_num_groups=8,
        attention_levels=(False, False, False, False, True),
        num_res_blocks=2,
        num_head_channels=8,
        use_flash_attention=True,
        cast_after_norm=True,
    )

def load_ae(weight_root, device):
    """MAISI autoencoder plus the latent (channels, shape) it produces for IMG_SHAPE."""
    ae = get_ae(Path(weight_root) / "MAISIv1", MAISI_CONFIG_FPATH, device)
    with amp_ctx(device), torch.inference_mode():
        z = ae.encode_stage_2_inputs(torch.zeros((1, 1, *IMG_SHAPE), device=device))
    latent_channels, latent_shape = z.shape[1], tuple(z.shape[2:])
    return ae, latent_channels, latent_shape

def build_model(strategy, weight_root, device):
    """
    Build the diffusion UNet (and MAISI autoencoder for LDM) and load EMA weights.
    Returns (px, ae, latent_channels, latent_shape); the last three are None
    for pixel-space strategies.
    """
    weight_root = Path(weight_root).resolve()
    ae, latent_channels, latent_shape = None, None, None
    if strategy == "ldm_rflow":
        ae, latent_channels, latent_shape = load_ae(weight_root, device)

    px = build_unet(strategy, latent_channels or 1)
    px.load_state_dict(load_file(weight_root / WEIGHT_FNAMES[strategy]))
    return px.to(device).eval(), ae, latent_channels, latent_shape

def to_np(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def to_nib_vol(x, affine=None, header=None):
    """
    Convert the volumetric numpy array to a NIFTI1 Image
    with the correct affine and orientation.
    This also quantizes to 16-bit unsigned integer.
    """
    if header is None:
        header = nib.Nifti1Header()
        header.set_data_shape(x.shape)
        header.set_zooms((1.0, 1.0, 1.0))
    else:
        header = header.copy()  # don't mutate the caller's header

    # Always overwrite, even if an input header was provided
    header.set_data_dtype(np.uint16)
    header.set_slope_inter(1, 0)
    header["descrip"] = "Synthetic image"

    if affine is None:
        affine = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    lo, hi = x.min(), x.max()
    int_x = ((x - lo) / max(hi - lo, 1e-8) * (2**16 - 1)).astype(np.uint16)
    return nib.Nifti1Image(int_x, affine=affine, header=header)