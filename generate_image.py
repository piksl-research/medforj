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
from tqdm.auto import tqdm

# ===== Deep learning =====
import torch
from torch.amp import GradScaler, autocast
from monai.networks.nets import DiffusionModelUNet
from monai.utils import set_determinism

# ===== Project-specific imports =====
from medforj.scheduler import *
from medforj.maisi_setup import get_ae, get_maisi_ldm, MaisiConditionedWrapper, get_maisi_scheduler

set_determinism(42)

def parse_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        print("GPU index not provided or no GPU support currently available.")
        print("!!! Running on CPU !!!")
        device = torch.device("cpu")
    return device

def to_np(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def sample(model, scheduler, device, noise=None,
           ae=None, latent_channels=None, latent_shape=None, verbose=True):
    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.float16):
        if ae is not None:
            if noise is None:
                noise = torch.randn((1, latent_channels, *latent_shape), device=device)
            # The MAISI autoencoder generates intensities in [0, 1]
            sample_img = scheduler.reverse_de(noise, model, verbose=verbose)
            sample_img = ae.decode_stage_2_outputs(sample_img)
            # Bring back to the image domain in [-1, 1]
            sample_img = sample_img * 2.0 - 1.0

        else:
            if noise is None:
                noise = torch.randn((1, 1, 192, 224, 192), device=device)
            sample_img = scheduler.reverse_de(noise, model, verbose=verbose)
    return sample_img

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
    parser.add_argument("--weight-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, default='flow')
    parser.add_argument("--ddim-steps", type=int, default=256)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    args.out_fpath.parent.mkdir(parents=True, exist_ok=True)
    device = parse_device(args.gpu_id)

    weight_root = args.weight_root.resolve()
    
    weight_fpaths = {
        "noise":     weight_root / "epsilon" / "model_epoch100.pt",
        "clean":     weight_root / "sample" / "model_epoch100.pt",
        "velocity":  weight_root / "velocity" / "model_epoch100.pt",
        "flow":      weight_root / "flow" / "model_epoch100.pt",
        "rflow":     weight_root / "rflow" / "model_epoch100.pt",
        "ldm_rflow": weight_root / "ldm_rflow" / "model_epoch100.pt",
    }
    
    MAISI_weight_dir = Path("../weights_archive/MAISIv1").resolve()
    MAISI_config_fpath = Path("./medforj/config_maisi.json").resolve()
            
    if args.strategy in ["noise", "clean", "velocity", "flow", "rflow"]:
        prediction_type = args.strategy
    elif args.strategy == "ldm_rflow":
        prediction_type = "rflow"

    if prediction_type == "ldm_rflow":
        ae = get_ae(MAISI_weight_dir, MAISI_config_fpath)
        ae = ae.to(device)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        
        with autocast(device_type="cuda", dtype=torch.float16), torch.inference_mode():
            dummy = torch.zeros((1, 1, 192, 224, 192), device=device)
            z_ldm = ae.encode_stage_2_inputs(dummy)
            latent_shape = tuple(z_ldm.shape[2:])
            latent_channels = z_ldm.shape[1]
    
        model = DiffusionModelUNet(
            spatial_dims=len(latent_shape),
            in_channels=latent_channels,
            out_channels=latent_channels,
            channels=(64, 128, 256, 512),
            norm_num_groups=8,
            attention_levels=(False, False, True, True),
            num_res_blocks=2,
            num_head_channels=32,
            use_flash_attention=True,
            cast_after_norm=True,
        ).to(device)
    else:
        ae = None
        latent_channels = None
        latent_shape = None
        model = DiffusionModelUNet(
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
        ).to(device)
    
        
    checkpoint_obj = torch.load(weight_fpaths[prediction_type], weights_only=True, map_location=device)
    model.load_state_dict(checkpoint_obj['model'])
    
    scheduler = DiffusionScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    scheduler.set_timesteps(num_inference_steps=args.ddim_steps)
    
    img = sample(model, scheduler, device, noise=None, 
                 ae=ae, latent_channels=latent_channels, latent_shape=latent_shape, 
                 verbose=args.verbose)

    to_nib_vol(to_np(img.float())).to_filename(args.out_fpath)
    
if __name__ == "__main__":
    main()
