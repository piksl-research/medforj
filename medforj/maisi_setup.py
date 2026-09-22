"""Load the MAISI v1 autoencoder, downloading its weights on first use."""
from pathlib import Path
import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from monai.bundle import ConfigParser

_REPO_ID = "nvidia/NV-Generate-CT"
_FILENAME = "models/autoencoder_v1.pt"


def get_ae(weight_dir, config_fpath, device=None) -> nn.Module:
    """Return the frozen MAISI v1 autoencoder.

    Weights are downloaded into ``weight_dir`` on the first call and reused afterwards.
    """
    weight_dir = Path(weight_dir).expanduser()
    ckpt_path = weight_dir / _FILENAME
    if not ckpt_path.exists():
        ckpt_path = Path(hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME, local_dir=weight_dir))

    with open(config_fpath, "r") as f:
        parser = ConfigParser(json.load(f))
    parser.parse(True)
    autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True)

    autoencoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    if device is not None:
        autoencoder = autoencoder.to(device)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    return autoencoder