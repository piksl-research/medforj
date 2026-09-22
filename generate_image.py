"""
This is a demo script to generate a single sample 
from a pretrained model and save it to disk as a 
NIFTI1 file in axial orientation.

This code is intended to be a template and can be
edited as needed for specific needs.
"""

import argparse
import torch
from medforj.cli_utils import setup, build_model, get_prediction_type, common_parser, to_np, to_nib_vol, amp_ctx
from medforj.scheduler import DiffusionScheduler

def sample(px, scheduler, device, noise=None,
           ae=None, latent_channels=None, latent_shape=None, verbose=True):
    with torch.inference_mode(), amp_ctx(device):
        if ae is not None:
            if noise is None:
                noise = torch.randn((1, latent_channels, *latent_shape), device=device)
            # The MAISI autoencoder generates intensities in [0, 1]
            sample_img = scheduler.reverse_de(noise, px, verbose=verbose)
            sample_img = ae.decode_stage_2_outputs(sample_img)
            # Bring back to the image domain in [-1, 1]
            sample_img = sample_img * 2.0 - 1.0

        else:
            if noise is None:
                noise = torch.randn((1, 1, 192, 224, 192), device=device)
            sample_img = scheduler.reverse_de(noise, px, verbose=verbose)
    return sample_img

def main(args=None):
    parser = argparse.ArgumentParser(parents=[common_parser()])
    args = parser.parse_args(args)
    
    device = setup(args)
    px, ae, latent_channels, latent_shape = build_model(args.strategy, args.weight_root, device)
    prediction_type = get_prediction_type(args.strategy)

    
    scheduler = DiffusionScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    scheduler.set_timesteps(num_inference_steps=args.n_steps)
    
    img = sample(px, scheduler, device, noise=None, 
                 ae=ae, latent_channels=latent_channels, latent_shape=latent_shape, 
                 verbose=args.verbose)

    to_nib_vol(to_np(img.float())).to_filename(args.out_fpath)
    
if __name__ == "__main__":
    main()
