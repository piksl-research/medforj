"""
Minimal training script for the MedForj priors.

Single GPU:
  python -m medforj.train --strategy flow --data-root DATA --out-root RUN --gpu-id 0
Multi-GPU (one node):
  torchrun --nproc_per_node=8 --standalone -m medforj.train --strategy flow \
      --data-root DATA --out-root RUN --batch-size 3

The released models used 8 GPUs x batch 3 (= 24), 100 epochs, Adam (lr 1e-4), bf16 autocast.
DATA should contain preprocess.py outputs. RUN can be passed directly as --weight-root
to generate_image.py and inverse_solve.py.
"""

import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, RandAffine, RandFlip, ScaleIntensity
from monai.utils import set_determinism
from safetensors.torch import save_file
from torch import multiprocessing as mp
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from medforj.cli_utils import WEIGHT_FNAMES, build_unet, get_prediction_type, load_ae
from medforj.scheduler import DiffusionScheduler

SEED = 0x42A42A42
EMA_MOMENTUM = 2e-4
ROTATE_DEG = 10.0
TRANSLATE_MM = 5.0  # voxels == mm at 1 mm isotropic

mp.set_sharing_strategy("file_system")  # avoids worker crashes with pinned memory


def setup_distributed(gpu_id):
    """Return (device, rank). Works with or without torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        return torch.device(f"cuda:{local_rank}"), dist.get_rank()
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}"), 0


def make_loader(data_root, batch_size, num_workers):
    """Every NIfTI under data_root, scaled to [-1, 1], with L-R flips and small rigid jitter."""
    transform = Compose([
        EnsureChannelFirst(),
        ScaleIntensity(minv=-1.0, maxv=1.0),
        RandFlip(spatial_axis=0, prob=0.5),
        RandAffine(
            rotate_range=[np.deg2rad(ROTATE_DEG)] * 3,
            translate_range=[TRANSLATE_MM] * 3,
            padding_mode="border",
            prob=0.5,
        ),
    ])
    dataset = ImageDataset(image_files=sorted(Path(data_root).rglob("*.nii*")), transform=transform)
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


@torch.no_grad()
def ema_update(ema, model):
    """ema <- (1 - m) * ema + m * model, once per optimizer step."""
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, EMA_MOMENTUM)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def save_checkpoint(out_root, strategy, epoch, model, ema, optimizer):
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "ema": ema.state_dict(),
         "optimizer": optimizer.state_dict(), "epoch": epoch},
        ckpt_dir / f"epoch_{epoch:03d}.pt",
    )
    # Inference-ready EMA weights: the same file generate_image.py loads.
    ema_state = {k: v.contiguous() for k, v in ema.state_dict().items()}
    save_file(ema_state, out_root / WEIGHT_FNAMES[strategy])


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, choices=list(WEIGHT_FNAMES))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1, help="per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", type=Path, help="checkpoint .pt to continue from")
    parser.add_argument("--gpu-id", type=int, default=0, help="ignored under torchrun")
    args = parser.parse_args(args)

    device, rank = setup_distributed(args.gpu_id)
    is_main = rank == 0
    set_determinism(SEED + rank)
    args.out_root.mkdir(parents=True, exist_ok=True)

    # ===== Autoencoder (LDM only): rank 0 downloads first, the others reuse the cache =====
    ae, latent_channels = None, 1
    if args.strategy == "ldm_rflow":
        if dist.is_initialized() and not is_main:
            dist.barrier()
        ae, latent_channels, _ = load_ae(args.out_root, device)
        if dist.is_initialized() and is_main:
            dist.barrier()

    # ===== Model, EMA, optimizer =====
    model = build_unet(args.strategy, latent_channels).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    train_model = DistributedDataParallel(model, device_ids=[device.index]) if dist.is_initialized() else model
    scheduler = DiffusionScheduler(prediction_type=get_prediction_type(args.strategy))
    loader, sampler = make_loader(args.data_root, args.batch_size, args.num_workers)

    # ===== Train =====
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", disable=not is_main, ncols=80)
        for i, batch in enumerate(pbar, start=1):
            x_0 = batch.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if ae is not None:
                    with torch.no_grad():  # MAISI expects intensities in [0, 1]
                        x_0 = ae.encode_stage_2_inputs(((x_0 + 1.0) / 2.0).clamp(0.0, 1.0))
                loss = scheduler.training_loss(train_model, x_0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_update(ema, model)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / i:.4f}")

        if is_main:
            save_checkpoint(args.out_root, args.strategy, epoch, model, ema, optimizer)
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()