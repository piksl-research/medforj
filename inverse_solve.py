"""
This is a demo script for inverse problem solving
using the MedForj weights. We provide the forward
operators used in the paper, but note that these
are for demonstration purposes. For your real-world
problem, please substitute for the physically accurate
forward problem appropriate for your data.

We assume the corrupted measurement `y` is an
axially oriented image with brain-centered FOV.

This code is intended to be a template and can be
edited as needed for specific needs.

vRAM appears to peak at around 14 GiB, with LDM on
nonlinear forward problems. A 16 GiB card is recommended.
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from medforj.cli_utils import setup, build_model, get_prediction_type, common_parser, to_np, to_nib_vol
from medforj.forward_ops import ForwardModel, Identity, SliceSelection, Masking, KSpaceMasking3D, MRIRigidMotion3D
from medforj.inverse_tools import DPSScheduler, ReSampleScheduler
from medforj.scheduler import DiffusionScheduler

def solve(px, y, forward_model, scheduler, device, noise=None,
          ae=None, latent_channels=None, latent_shape=None, verbose=True):
    if ae is not None:
        if noise is None:
            noise = torch.randn((1, latent_channels, *latent_shape), device=device)
        z_hat = scheduler.reverse_de(noise, y, forward_model, px, verbose=verbose)
        x_hat = scheduler.decode(z_hat).float()
        # Bring back to the image domain in [-1, 1]
        x_hat = (x_hat * 2.0 - 1.0).clamp(-1.0, 1.0)
    else:
        if noise is None:
            noise = torch.randn((1, 1, 192, 224, 192), device=device)
        x_hat = scheduler.reverse_de(noise, y, forward_model, px, eta=1.0, verbose=verbose).float()
    return x_hat

def main(args=None):
    parser = argparse.ArgumentParser(parents=[common_parser()])
    parser.add_argument("--inp-fpath", type=Path, required=True)
    parser.add_argument("--task", required=True,
                    choices=["slice_selection", "inpainting", "rician_denoising", "kspace_accel", "motion"])
    args = parser.parse_args(args)
    
    device = setup(args)
    px, ae, latent_channels, latent_shape = build_model(args.strategy, args.weight_root, device)
    prediction_type = get_prediction_type(args.strategy)

    # ===== Load x and check that it's been preprocessed correctly =====
    obj = nib.load(args.inp_fpath)
    if obj.shape != (192, 224, 192):
        raise ValueError(
            f"Input image has shape {obj.shape}, expected (192, 224, 192). "
            f"Please run preprocess.py on '{args.inp_fpath}' first."
        )
    x = torch.from_numpy(obj.get_fdata(dtype=np.float32)).unsqueeze(0).unsqueeze(1).to(device)
    x -= x.min()
    x /= x.max()
    x *= 2
    x -= 1
    
    # ===== Simulate y based on the task =====

    # Forward models are the deterministic A and a noise model
    # In this paper, non-Rician tasks use a "No noise Gaussian model"    
    match args.task:
        case "slice_selection":
            # This is a demo script. Feel free to change slice thickness and separation
            # as well as the axis, based on your preferences. If you choose the A-P axis
            # make sure to update `hr_shape` as well.
            A = SliceSelection(hr_shape=192, hr_spacing_mm=1, slice_thickness_mm=7, 
                               slice_separation_mm=9.5, axis=2, device=device)
            forward_model = ForwardModel(A)
            zeta = 20
        case "inpainting":
            mask = torch.ones(1, 1, 192, 224, 192, device=device).float()
            mask[..., 64:-64, 64:-64, 64:-64] = 0 # Feel free to make any mask you wish
            A = Masking(mask)
            forward_model = ForwardModel(A)
            zeta = 20
        case "rician_denoising":
            A = Identity()
            forward_model = ForwardModel(A, "rician", sigma=0.1)
            zeta = 5e-4 # Rician NLL is on a different scale than usual DPS solving
        case "kspace_accel":
            A = KSpaceMasking3D(acceleration=12, center_fraction=0.08, power=4.0)
            forward_model = ForwardModel(A)
            zeta = 20
        case "motion":
            A = MRIRigidMotion3D(degrees=(6, 6, 6), translation_mm=(6, 6, 6), time=0.45).to(device)
            forward_model = ForwardModel(A)
            zeta = 20

    # ===== Inverse problem solver scheduler setup =====
    if args.strategy in ["noise", "velocity"] and args.task != "rician_denoising":
        zeta = 500 # empirical hyperparameter, sorry

    if args.strategy == "ldm_rflow":
        scheduler = ReSampleScheduler(ae=ae, cg_iters=20)
        scheduler.set_timesteps(num_inference_steps=args.n_steps)
    else:
        scheduler = DPSScheduler(num_train_timesteps=1000, prediction_type=prediction_type, zeta=zeta)
        scheduler.set_timesteps(num_inference_steps=args.n_steps)

    # ===== Run both forward and inverse, then save =====
    y = forward_model(x)
    x_hat = solve(px, y, forward_model, scheduler, device, 
                  ae=ae, latent_channels=latent_channels, latent_shape=latent_shape,
                  verbose=args.verbose)
    to_nib_vol(to_np(x_hat.float()), affine=obj.affine, header=obj.header).to_filename(args.out_fpath)

if __name__ == "__main__":
    main()