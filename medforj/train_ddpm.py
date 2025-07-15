# ===== Standard imports =====
import argparse
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import torch.distributed as dist
# ===== Deep learning =====
import wandb
from monai.data import DataLoader, ImageDataset
from monai.inferers import DiffusionInferer
from monai.losses import SSIMLoss
from monai.networks.nets import DiffusionModelUNet
from monai.transforms import (
    Transform,
    Compose,
    RandAffine,
    RandFlip,
    ScaleIntensity,
    EnsureChannelFirst,
)
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# ===== Project-specific imports =====
from .plotting import lightbox, to_np
from .scheduler import DDIMScheduler, DDIMPredictionType

# Set deterministic training for reproducibility
set_determinism(42)

# Environment variable for parallelism
os.environ["NCCL_P2P_LEVEL"] = "NVL"

mpl.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
    }
)


class MRIDiffusion:
    def __init__(
        self,
        device_id,
        prediction_type: str = "epsilon",
        num_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        num_norm_groups: int = 8,
        attention_levels: tuple[bool, ...] = (False, False, False, False, True),
        num_res_blocks: int = 2,
        num_head_channels: int = 8,
        image_size: tuple[int, ...] = (192, 224, 192),
        checkpoint: Path = None,
    ):

        self.device = torch.device(f"cuda:{device_id}")

        self.amp_enabled = None
        self.clip_grad_norm = None

        # Save parameters for wandb logging
        self.num_channels = num_channels
        self.num_norm_groups = num_norm_groups
        self.attention_levels = attention_levels
        self.num_res_blocks = num_res_blocks
        self.num_head_channels = num_head_channels
        self.prediction_type = prediction_type
        self.image_size = image_size

        model = DiffusionModelUNet(
            spatial_dims=len(self.image_size),
            in_channels=1,
            out_channels=1,
            channels=self.num_channels,
            norm_num_groups=self.num_norm_groups,
            attention_levels=self.attention_levels,
            num_res_blocks=self.num_res_blocks,
            num_head_channels=self.num_head_channels,
        )
        model.to(self.device)

        self.optimizer, self.scaler = None, None

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            prediction_type=self.prediction_type,
            clip_sample=True,
            clip_sample_min=-1,
            clip_sample_max=1,
        )
        self.scheduler.set_timesteps(num_inference_steps=64)
        self.inferer = DiffusionInferer(self.scheduler)

        self.checkpoint = checkpoint
        self.checkpoint_obj = None
        if self.checkpoint is not None:
            self.checkpoint_obj = torch.load(self.checkpoint)
            model.load_state_dict(self.checkpoint_obj["model"])

        # Distributed data parallel model replaces the base `self.model`
        self.model = model
        if dist.is_initialized():
            self.model = DistributedDataParallel(model, device_ids=[device_id])

        self.start_epoch = 0

    @staticmethod
    def create_dataloader(
        dataset_dir,
        batch_size,
        minv,
        maxv,
        num_workers=8,
        prefetch_factor=8,
        use_rand_affine=True,
    ):
        transforms: list[Transform] = [
            EnsureChannelFirst(),
            ScaleIntensity(minv=minv, maxv=maxv),
            RandFlip(spatial_axis=0, prob=0.5),
        ]
        if use_rand_affine:
            transforms.append(
                RandAffine(
                    rotate_range=[
                        (-np.deg2rad(5), np.deg2rad(5)),
                        (-np.deg2rad(5), np.deg2rad(5)),
                        (-np.deg2rad(5), np.deg2rad(5)),
                    ],
                    translate_range=[
                        (-5, 5),
                        (-5, 5),
                        (-5, 5),
                    ],
                    padding_mode="border",
                    prob=0.5,
                )
            )

        train_dataset = ImageDataset(
            image_files=tuple(dataset_dir.rglob("*.nii*")),
            transform=Compose(transforms),
        )

        sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )

        return train_loader, sampler

    def train(
        self,
        project,
        exp_name,
        output_root,
        dataset_path,
        batch_size,
        num_epochs,
        base_lr=1e-4,
        new_wandb_run=False,
        num_workers=8,
        prefetch_factor=8,
        disable_amp=False,
        clip_grad_norm=None,
        disable_rand_affine=False,
    ):
        models_savedir = output_root / project / exp_name / "models"
        models_savedir.mkdir(exist_ok=True, parents=True)

        self.amp_enabled = not disable_amp
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=base_lr)
        self.scaler = GradScaler(enabled=self.amp_enabled)
        if self.checkpoint_obj is not None:
            self.optimizer.load_state_dict(self.checkpoint_obj["optimizer"])
            self.scaler.load_state_dict(self.checkpoint_obj["scaler"])
            self.start_epoch = self.checkpoint_obj["epoch"]

        self.start_epoch += 1

        train_loader, sampler = self.create_dataloader(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            minv=-1,
            maxv=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            use_rand_affine=not disable_rand_affine,
        )

        run_id = None
        if self.checkpoint is not None and not new_wandb_run:
            run_id = self.checkpoint_obj["run_id"]

        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.init(
                project=project,
                name=exp_name,
                dir=str(models_savedir.parent),
                id=run_id,
                resume="allow",
                config={
                    "dataset": str(dataset_path),
                    "num_channels": self.num_channels,
                    "num_norm_groups": self.num_norm_groups,
                    "attention_levels": self.attention_levels,
                    "num_res_blocks": self.num_res_blocks,
                    "num_head_channels": self.num_head_channels,
                    "prediction_type": self.prediction_type,
                    "image_size": self.image_size,
                    "batch_size": batch_size,
                    "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                    "lr": base_lr,
                    "num_epochs": num_epochs,
                    "disable_amp": disable_amp,
                    "clip_grad_norm": clip_grad_norm,
                },
            )

        for epoch in range(self.start_epoch, num_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = self.epoch(train_loader, epoch)

            if dist.is_initialized():
                dist.barrier()
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.save_model(models_savedir / f"model_epoch{epoch}.pt", epoch, epoch_loss)
                self.wandb_plot(epoch_loss, epoch)
            if dist.is_initialized():
                dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.finish()

    def epoch(self, data_loader, epoch):
        if not dist.is_initialized() or dist.get_rank() == 0:
            loader = tqdm(enumerate(data_loader), total=len(data_loader), ncols=70)
            loader.set_description(f"Epoch {epoch:03d}")
        else:
            loader = enumerate(data_loader)

        epoch_loss = {"loss": 0}
        step = 0
        for step, batch in loader:
            loss_dict = self.step(batch.to(self.device))

            if not dist.is_initialized() or dist.get_rank() == 0:
                for key in loss_dict:
                    epoch_loss[key] += loss_dict[key]
                loader.set_postfix({k: v / (step + 1) for k, v in epoch_loss.items()})

        # Clear memory
        del batch
        torch.cuda.empty_cache()

        return {k: v / (step + 1) for k, v in epoch_loss.items()}

    def step(self, images):
        self.model.train()
        self.optimizer.zero_grad()
        batch_size = images.shape[0]

        with autocast(device_type="cuda", enabled=self.amp_enabled):
            # Generate random noise
            noise = torch.randn_like(images).to(self.device)

            # Create timesteps
            timesteps = torch.randint(
                0,
                self.inferer.scheduler.num_train_timesteps,
                (batch_size,),
                device=images.device,
            ).long()

            if self.prediction_type == DDIMPredictionType.EPSILON:
                target = noise
            elif self.prediction_type == DDIMPredictionType.SAMPLE:
                target = images
            elif self.prediction_type == DDIMPredictionType.VELOCITY:
                target = self.inferer.scheduler.get_velocity(images, noise, timesteps)
            elif self.prediction_type == DDIMPredictionType.FLOW:
                target = noise - images
            else:
                raise ValueError("Invalid prediction type")

            # Send batch to time t in the diffusion trajectory
            noisy_image = self.scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps
            )
            pred = self.model(x=noisy_image, timesteps=timesteps)

            loss = functional.mse_loss(pred, target)

        self.scaler.scale(loss).backward()
        if self.clip_grad_norm is not None:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if self.ssim_loss_enabled:
                dist.all_reduce(ssim_loss, op=dist.ReduceOp.AVG)
            if self.l1_loss_enabled:
                dist.all_reduce(l1_loss, op=dist.ReduceOp.AVG)

        losses = {
            "loss": loss.item(),
        }
        return losses

    def gen_training_sample(self):
        with torch.inference_mode(), autocast(device_type="cuda", enabled=True):
            samples_ = []
            for _ in range(4):
                noise = torch.randn(
                    (1, 1) + self.image_size,
                    device=self.device,
                )

                # noinspection PyTypeChecker
                sample = self.inferer.sample(
                    input_noise=noise,
                    diffusion_model=self.model,
                    verbose=False,
                )
                samples_.append(sample)

            sample_coefs = torch.cat(samples_, dim=0)
            sample_img = sample_coefs.squeeze()
        return sample_img

    def wandb_plot(self, loss, step):
        sample_imgs = self.gen_training_sample()
        sample_pngs = [wandb.Image(lightbox(to_np(sample))) for sample in sample_imgs]
        wandb.log(
            loss | {"samples": sample_pngs},
            step=step,
            commit=True,
        )
        # Clear memory
        del sample_imgs
        torch.cuda.empty_cache()

    def save_model(self, fpath, epoch, epoch_loss):
        save_dict = {
            "model": (
                self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict()
            ),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "run_id": wandb.run.id,
        }
        torch.save(save_dict, fpath)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument(
        "--prediction-type",
        type=str,
        default="flow",
        choices=["epsilon", "sample", "velocity", "flow"],
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
    )
    parser.add_argument("--num-norm-groups", type=int, default=8)
    parser.add_argument(
        "--attention-levels",
        type=bool,
        nargs="+",
        default=[False, False, False, False, True],
    )
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-head-channels", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs="+", default=[192, 224, 192])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--new-wandb-run", action="store_true", default=False)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--disable-rand-affine", action="store_true", default=False)
    parsed = parser.parse_args(args)

    # Parallelism setup
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        device_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device_id)
        dist.init_process_group("nccl")
    else:
        if parsed.gpu_id is None:
            raise ValueError("Either use distributed (torchrun) or provide --gpu-id.")
        device_id = int(parsed.gpu_id)
        torch.cuda.set_device(device_id)
    print(f"Initializing job on gpu {device_id}.")

    diff_model = MRIDiffusion(
        device_id=device_id,
        prediction_type=parsed.prediction_type,
        num_channels=tuple(parsed.num_channels),
        num_norm_groups=parsed.num_norm_groups,
        attention_levels=tuple(parsed.attention_levels),
        num_res_blocks=parsed.num_res_blocks,
        num_head_channels=parsed.num_head_channels,
        image_size=tuple(parsed.image_size),
        checkpoint=parsed.checkpoint,
    )

    diff_model.train(
        project=parsed.project,
        exp_name=parsed.exp_name,
        output_root=parsed.output_root,
        dataset_path=parsed.dataset_path,
        batch_size=parsed.batch_size,
        num_epochs=parsed.num_epochs,
        new_wandb_run=parsed.new_wandb_run,
        base_lr=parsed.base_lr,
        num_workers=parsed.num_workers,
        prefetch_factor=parsed.prefetch_factor,
        disable_amp=parsed.disable_amp,
        clip_grad_norm=parsed.clip_grad_norm,
        disable_rand_affine=parsed.disable_rand_affine,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
