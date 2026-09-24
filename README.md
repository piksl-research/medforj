# MedForj: Diffusion-Driven Generation of Minimally Preprocessed Brain MRI
by [PIKSL](https://piksl-research.github.io/) and [IACL](https://iacl.ece.jhu.edu/).

This repo contains minimal training and inference code for 3D diffusion models with various strategies (noise prediction, clean prediction, velocity, flow, rectified flow, LDM rectified flow) leveraging the [MONAI](https://monai.io/) framework and some custom adjustments.

## Quick-start

We recommend using `uv` for installation to automatically handle torch and driver compat.
```
git clone https://github.com/piksl-research/medforj.git
cd medforj

# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## Strategies and weights

Each strategy is a separately trained prior. Pass it with `--strategy` and make sure the matching file is in `--weight-root`.

| `--strategy` | Model predicts | Space | Weight file |
|---|---|---|---|
| `noise`     | the added noise ε                  | image  | `MedForj-weights-noise_ema.safetensors` |
| `clean`     | the clean image x₀                 | image  | `MedForj-weights-clean_ema.safetensors` |
| `velocity`  | velocity v = √ᾱ·ε − √(1−ᾱ)·x₀      | image  | `MedForj-weights-velocity_ema.safetensors` |
| `flow`      | flow ε − x₀                        | image  | `MedForj-weights-flow_ema.safetensors` |
| `rflow`     | rectified flow x₀ − ε              | image  | `MedForj-weights-rflow_ema.safetensors` |
| `ldm_rflow` | rectified flow x₀ − ε              | LDM | `MedForj-weights-ldm_rflow_ema.safetensors` |


Download the weights from HuggingFace into one folder `/PATH/TO/WEIGHTS`:
[https://huggingface.co/piksl-research/medforj-brain-t1w-3d](https://huggingface.co/piksl-research/medforj-brain-t1w-3d)

This CLI command can also do it:
```
huggingface-cli download piksl-research/medforj-brain-t1w-3d --local-dir /PATH/TO/WEIGHTS
```

`--weight-root` should then look like this:
```
/PATH/TO/WEIGHTS/
├── MedForj-weights-noise_ema.safetensors
├── MedForj-weights-clean_ema.safetensors
├── MedForj-weights-velocity_ema.safetensors
├── MedForj-weights-flow_ema.safetensors
├── MedForj-weights-rflow_ema.safetensors
├── MedForj-weights-ldm_rflow_ema.safetensors
└── MAISIv1/                      # ldm_rflow only; downloaded automatically on first use
    └── models/autoencoder_v1.pt  # NVIDIA MAISI v1 autoencoder
```

## Using MedForj
Sample an image from the pre-trained weights:
```
python generate_image.py --out-fpath /PATH/TO/OUTPUT/my-new-image.nii.gz --weight-root /PATH/TO/WEIGHTS/ --strategy flow --gpu-id 0 --verbose
```

Inverse problem solving requires preprocessing first, then simulates the corrupted image `y` before estimating the restored image `x_hat`:
```
python preprocess.py --inp-fpath raw_t1w.nii.gz --out-fpath prep.nii.gz [--mask-fpath brain_mask.nii.gz]

python inverse_solve.py --inp-fpath prep.nii.gz --task slice_selection \
    --out-fpath x_hat.nii.gz --y-fpath y.nii.gz --weight-root /PATH/TO/WEIGHTS --strategy flow --gpu-id 0
```

Valid tasks: `slice_selection`, `inpainting`, `rician_denoising`, `kspace_accel`, `motion`.


## Training your own model

All data should be within (sub)folders in a particular directory. 

### Single GPU:

If your GPU is large enough, you can run:
```
python -m medforj.train --strategy flow --data-root DATA --out-root RUN --gpu-id 0
```

### Multi-GPU (one node):

We support PyTorch DDP. Use this command:

```
torchrun --nproc_per_node=8 --standalone -m medforj.train --strategy flow \
      --data-root DATA --out-root RUN --batch-size 3
```

# Citation
If you find our code or models useful in your work, please consider citing them as:

```
@misc{medforj-t1,
      title={Diffusion-Driven Generation of Minimally Preprocessed Brain MRI},
      author={Samuel W. Remedios and Aaron Carass and Jerry L. Prince and Blake E. Dewey and others},
      year={2026},
      eprint={todo},
      url={todo},
}
```
