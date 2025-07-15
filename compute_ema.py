"""
Usually, EMA is computed on-the-fly. In this project, we
computed EMA afterwards. They are equivalent, but this
requires access to all weights over time.
"""
import numpy as np
import os
from time import time
from tqdm.auto import tqdm
from pathlib import Path
import torch
from monai.networks.nets import DiffusionModelUNet
from copy import deepcopy

weights_folder = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2])

weight_files = [os.path.join(weights_folder, f) 
                for f in os.listdir(weights_folder) if f.endswith(".pt")]
weight_files = sorted(weight_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

ema_model = None
momentum = 0.1

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    norm_num_groups=8,
    attention_levels=(False, False, False, False, True),
    num_res_blocks=2,
    num_head_channels=8,
)

for i, weight_path in tqdm(enumerate(weight_files), total=len(weight_files)):

    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint["model"])

    with torch.no_grad():
        if ema_model is None:
            ema_model = deepcopy(model)
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(1.0 - momentum).add_(model_p.data, alpha=momentum)
        for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
            ema_b.data = model_b.data

torch.save({"model": ema_model.state_dict()}, output_path)
print(f"EMA weights saved to: {output_path}")
