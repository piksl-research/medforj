# MedForj: Diffusion-Driven Generation of Minimally Preprocessed Brain MRI
by [PIKSL](https://piksl-research.github.io/) and [IACL](https://iacl.ece.jhu.edu/).

This repo contains minimal training and inference code for 3D DDPMs with various prediction types (e.g., sample, velocity, and flow) leveraging the [MONAI](https://monai.io/) framework and some custom adjustments.

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
python demo.py --out-fpath {MY_IMG.nii.gz} --weight-fpath {WEIGHTS.pt} --prediction-type {sample/velocity/flow} --gpu-id 0
```

Choose the `prediction_type` according to the pre-trained weights used.

The EMA weights are available on HuggingFace:
[https://huggingface.co/piksl-research/medforj-brain-t1w-3d](https://huggingface.co/piksl-research/medforj-brain-t1w-3d)


## Training your own model

Our code uses [Weights and Biases](https://wandb.ai/) for tracking metrics. You can follow their API to disable WandB for your training if you wish. Otherwise, first ensure to log in with your own WandB account.

All data should be within (sub)folders in a particular directory. 

### For single-GPU machines:

If your GPU is large enough, you can run:
```
python -m medforj.train_ddpm --project {YOUR_WANDB_PROJECT} --exp-name {RUN_NAME} --dataset-path {PATH/TO/YOUR/DATA} --output-root {PATH/TO/YOUR/WEIGHTS} --num-epochs 1000 --batch-size 2 --prediction-type flow  --num-workers 12 --disable-amp 
```


### For multi-GPU machines:

We support PyTorch DDP. Use this command:

```
torchrun --nproc_per_node=4 --standalone -m medforj.train_ddpm --project {YOUR_WANDB_PROJECT} --exp-name {RUN_NAME} --dttaset-path {PATH/TO/YOUR/DATA} --output-root {PATH/TO/YOUR/WEIGHTS} --num-epochs 1000 --batch-size 2 --prediction-type flow  --num-workers 12 --disable-amp
```

# Citation
If you find our code or models useful in your work, please consider citing them as:

```
@misc{medforj-t1,
      title={Diffusion-Driven Generation of Minimally Preprocessed Brain MRI},
      author={Samuel W. Remedios and Aaron Carass and Jerry L. Prince and Blake E. Dewey and others},
      year={2025},
      eprint={todo},
      url={todo},
}
```
