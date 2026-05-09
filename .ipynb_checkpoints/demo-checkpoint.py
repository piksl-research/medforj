"""
This is a demo script to generate a single sample 
from a pretrained model and save it to disk as a 
NIFTI1 file in axial orientation.

This code is intended to be a template and can be
edited as needed for specific needs.
"""

# ===== Standard imports =====
import argparse
import numpy as np
import sys
import nibabel as nib
from pathlib import Path

# ===== Deep learning =====
import torch
from torch.amp import GradScaler, autocast
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.utils import set_determinism

# ===== Project-specific imports =====
from medforj.scheduler import *

set_determinism(42)

def sample(model, inferer, device, noise=None, verbose=True):
    with torch.inference_mode(), autocast(device_type="cuda", enabled=True):
        if noise is None:
            noise = torch.randn((1, 1, 192, 224, 192), device=device)
        img = inferer.sample(input_noise=noise[i:i+1], diffusion_model=model, verbose=verbose)
    return img.squeeze()

def to_nib_vol(x):
    """
    Convert the volumetric numpy array to a NIFTI1 Image
    with the correct affine and orientation.
    This also quantizes to 16-bit unsigned integer.
    """
    header = nib.Nifti1Header()
    header.set_data_shape(x.shape)
    header.set_data_dtype(np.uint16)
    header.set_zooms((1.0, 1.0, 1.0))
    header['descrip'] = "Synthetic image"

    affine = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    int_x = ((x - x.min()) / (x.max() - x.min()) * (2**16 - 1)).astype(np.uint16)
    return nib.Nifti1Image(int_x, affine=affine, header=header)

def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-fpath", type=Path, required=True)
    parser.add_argument("--weight-fpath", type=Path, required=True)
    parser.add_argument("--prediction-type", type=str, required=True)
    parser.add_argument("--ddim-steps", type=int, default=64)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    args.out_fpath.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}")

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        norm_num_groups=8,
        attention_levels=(False, False, False, False, True),
        num_res_blocks=2,
        num_head_channels=8,
    ).to(device)

    checkpoint_obj = torch.load(args.weight_fpath, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint_obj['model'])

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        prediction_type=args.prediction_type,
        clip_sample=True,
        clip_sample_min=-1,
        clip_sample_max=1,
    )
    scheduler.set_timesteps(num_inference_steps=args.ddim_steps)
    inferer = DiffusionInferer(scheduler)

    img = to_np(sample(model, inferer, device, verbose=args.verbose))
    to_nib_vol(img).to_filename(out_fpath)
