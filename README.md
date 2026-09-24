# MedForj: Diffusion-Driven Generation of Minimally Preprocessed Brain MRI
by [PIKSL](https://piksl-research.github.io/) and [IACL](https://iacl.ece.jhu.edu/).

This repo contains minimal training and inference code for 3D diffusion models with various strategies (noise prediction, clean prediction, velocity, flow, rectified flow, LDM rectified flow) leveraging the [MONAI](https://monai.io/) framework and some custom adjustments.

## Quick-start

Clone the repository, create a new virtual environment (e.g., with miniconda), and pip install the required libraries:
```
cd $HOME && git clone https://github.com/piksl-research/medforj.git
cd $HOME/medforj
conda create -n medforj python==3.11
conda activate medforj
pip install .
```

Sample an image from the pre-trained weights:
```
python generate_image.py --out-fpath /PATH/TO/OUTPUT/my-new-image.nii.gz --weight-root /PATH/TO/WEIGHTS/ --strategy STRAT --gpu-id 0 --verbose
```

Inverse problem solve:
```
python inverse_solve.py --strategy STRAT {--more details...}
```

Choose the `strategy` according to the pre-trained weights used.

The EMA weights are available on HuggingFace:
[https://huggingface.co/piksl-research/medforj-brain-t1w-3d](https://huggingface.co/piksl-research/medforj-brain-t1w-3d)


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
