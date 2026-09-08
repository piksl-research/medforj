from pathlib import Path
import json
import argparse
import os
import shutil
import torch
import torch.nn as nn

from monai.bundle import ConfigParser
from monai.apps import download_url
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download

def get_maisi_scheduler(config_fpath: Path):
    args = argparse.Namespace()

    with open(config_fpath, "r") as f:
        config_dict = json.load(f)

    for k, v in config_dict.items():
        setattr(args, k, v)

    return define_instance(args, "noise_scheduler")

class MaisiConditionedWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        modality: int = 9,
        spacing=(1.0, 1.0, 1.0),
        top_region_index=(0, 1, 0, 0),
        bottom_region_index=(0, 0, 1, 0),
    ):
        super().__init__()
        self.model = model
        self.modality = modality
        self.spacing = spacing
        self.top_region_index = top_region_index
        self.bottom_region_index = bottom_region_index

    def forward(self, x, timesteps, context=None):
        b = x.shape[0]
        device = x.device

        class_labels = torch.full(
            (b,),
            self.modality,
            device=device,
            dtype=torch.long,
        )

        spacing_tensor = torch.tensor(
            self.spacing,
            device=device,
            dtype=x.dtype,
        ).view(1, 3).repeat(b, 1)

        top_region_index_tensor = torch.tensor(
            self.top_region_index,
            device=device,
            dtype=x.dtype,
        ).view(1, 4).repeat(b, 1)

        bottom_region_index_tensor = torch.tensor(
            self.bottom_region_index,
            device=device,
            dtype=x.dtype,
        ).view(1, 4).repeat(b, 1)

        return self.model(
            x,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )

        

# From https://github.com/Project-MONAI/tutorials/blob/main/generation/maisi/scripts/download_model_data.py
def fetch_to_hf_path_cmd(
    items: List[Dict[str, str]],
    root_dir: str = "./",  # (kept for signature compatibility; not required)
    revision: str = "main",
    overwrite: bool = False,
    token: Optional[str] = None,  # or rely on env HF_TOKEN / HUGGINGFACE_HUB_TOKEN
) -> list[str]:
    """
    items: list of {"repo_id": "...", "filename": "path/in/repo.ext", "path": "local/target.ext"}
    Returns list of saved local paths (in the same order as items).

    Pure Python implementation (CI-safe): no `huggingface-cli` dependency.
    """
    saved = []

    for it in items:
        repo_id = it["repo_id"]
        repo_file = it["filename"]
        dst = Path(it["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            saved.append(str(dst))
            continue

        # Download into HF cache, then copy to requested destination
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            revision=revision,
            token=token,  # if None, huggingface_hub will use env / cached auth if present
        )

        # Copy/move into place
        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(cached_path, dst)
        saved.append(str(dst))

    return saved


def define_instance(args, instance_def_key):
    """
    Define and instantiate an object based on the provided arguments and instance definition key.

    This function uses a ConfigParser to parse the arguments and instantiate an object
    defined by the instance_def_key.

    Args:
        args: An object containing the arguments to be parsed.
        instance_def_key (str): The key used to retrieve the instance definition from the parsed content.

    Returns:
        The instantiated object as defined by the instance_def_key in the parsed configuration.
    """
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

def get_ae(weight_dir, config_fpath):

    # Hard-coded, from official repo
    # "v1 is for commercial purposes; v2 trained on more data"
    file = {
        "path": weight_dir / "autoencoder_v1.pt",
        "repo_id": "nvidia/NV-Generate-CT",
        "filename": "models/autoencoder_v1.pt",
    }

    # file = {
    #     "path": weight_dir / "autoencoder_v2.pt",
    #     "repo_id": "nvidia/NV-Generate-MR",
    #     "filename": "models/autoencoder_v2.pt",
    # }

    if not file["path"].exists():
        path = fetch_to_hf_path_cmd([file], root_dir=weight_dir, revision="main")
        print("saved to:", path)
    
    args = argparse.Namespace()
    
    with open(config_fpath, "r") as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        setattr(args, k, v)
    
    autoencoder = define_instance(args, "autoencoder_def")
    
    checkpoint_autoencoder = torch.load(file["path"], weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    
    return autoencoder


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def _load_state_dict_flexible(model: nn.Module, ckpt_path: Path, map_location="cpu"):
    """
    MAISI diffusion checkpoints may contain MONAI MetaTensor objects.
    Try weights_only=True first; if that fails, retry standard torch.load.
    Only use the fallback for trusted checkpoints.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed for {ckpt_path}: {e}")
        print("Retrying with weights_only=False; do this only for trusted checkpoints.")
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Most MAISI files appear to be direct state_dicts, but this makes it robust.
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "network", "diffusion_unet"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # Strip common DDP/module prefixes if present.
    if isinstance(ckpt, dict):
        ckpt = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in ckpt.items()
        }

    checkpoint = torch.load(
        "/iacl/pg23/sam/weights_archive/MAISIv1/diff_unet_3d_rflow-mr-brain_v0.pt",
        map_location="cpu",
        weights_only=False,  # trusted NVIDIA checkpoint
    )
    
    print(checkpoint.keys())
    
    state_dict = checkpoint["unet_state_dict"]
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    print("first missing:", missing[:20])
    print("first unexpected:", unexpected[:20])

    if missing or unexpected:
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        if len(missing) < 20:
            print("Missing:", missing)
        if len(unexpected) < 20:
            print("Unexpected:", unexpected)

    return model


def get_maisi_ldm(
    weight_dir: Path,
    config_fpath: Path,
    version: str = "rflow",   # "ddpm" for MAISIv1-style, or "rflow" for newer rectified-flow
    device=None,
):
    weight_dir = Path(weight_dir)
    weight_dir.mkdir(parents=True, exist_ok=True)

    if version == "ddpm":
        file = {
            "path": weight_dir / "diff_unet_3d_ddpm-mr-brain_v0.pt",
            "repo_id": "nvidia/NV-Generate-MR-Brain",
            "filename": "models/diff_unet_3d_ddpm-mr-brain_v0.pt",
        }
    elif version == "rflow":
        file = {
            "path": weight_dir / "diff_unet_3d_rflow-mr-brain_v0.pt",
            "repo_id": "nvidia/NV-Generate-MR-Brain",
            "filename": "models/diff_unet_3d_rflow-mr-brain_v0.pt",
        }
    else:
        raise ValueError(f"Unknown version={version!r}; expected 'ddpm' or 'rflow'.")

    if not file["path"].exists():
        path = fetch_to_hf_path_cmd([file], root_dir=str(weight_dir), revision="main")
        print("saved to:", path)

    args = argparse.Namespace()
    with open(config_fpath, "r") as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        setattr(args, k, v)

    diffusion_unet = define_instance(args, "diffusion_unet_def")
    diffusion_unet = _load_state_dict_flexible(diffusion_unet, file["path"], map_location="cpu")

    if device is not None:
        diffusion_unet = diffusion_unet.to(device)

    diffusion_unet.eval()
    for p in diffusion_unet.parameters():
        p.requires_grad_(False)

    return diffusion_unet