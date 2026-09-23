"""

Minimal brain MRI preprocessing.

Steps:
  1. Brain mask: use the one provided, or predict one with HD-BET.
  2. Resample any axis finer than 1 mm to 1 mm (important: this script is not intended for images worse than 1mm on any axis).
  3. Reorient image to LPS and mask.
  4. Clip intensities to the 0.1-99.9th percentiles and rescale to uint16.
  5. Crop/pad to a 192 x 224 x 192 mm field of view centered on the brain.

The mask is only used to position the crop; it is NOT applied to the image.

Usage:
  python preprocess.py input.nii.gz output.nii.gz [--mask brain_mask.nii.gz]
"""
import argparse
import math
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from radifox.utils.resize.affine import update_affine
from radifox.utils.resize.scipy import resize
from scipy.ndimage import label

MIN_RES_MM = 1.0
TARGET_FOV_MM = (192, 224, 192)
P_LOW, P_HIGH = 0.1, 99.9


def hdbet_mask(image_path, device="cuda"):
    """Predict a brain mask with HD-BET (model weights download on first use)."""
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=True, device=torch.device(device))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "brain.nii.gz"
        hdbet_predict(str(image_path), str(out), predictor,
                      keep_brain_mask=True, compute_brain_extracted_image=False)
        m = nib.load(Path(tmp) / "brain_bet.nii.gz")  # HD-BET's mask filename
        return nib.Nifti1Image(m.get_fdata(dtype=np.float32), m.affine)


def resample(img, dxyz, order):
    """Resample by per-axis factors (<1 upsamples, >1 downsamples)."""
    x = img.get_fdata(dtype=np.float32)
    y = resize(x, dxyz=dxyz, order=order)
    y = np.clip(y, x.min(), x.max())  # remove cubic-spline overshoot
    return nib.Nifti1Image(y, update_affine(img.affine, dxyz))


def to_lps(img):
    """Reorient an image to LPS voxel storage order."""
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt("LPS"),
    )
    return img.as_reoriented(ornt)


def crop_pad_to_brain(x, mask, target_shape):
    """Crop/pad x to target_shape, centered on the largest connected mask component."""
    labels, _ = label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    coords = np.argwhere(labels == sizes.argmax())
    center = (coords.min(axis=0) + coords.max(axis=0)) // 2

    crop, pad = [], []
    for c, t, n in zip(center, target_shape, x.shape):
        start = c - (t - 1) // 2
        stop = start + t
        crop.append(slice(max(start, 0), min(stop, n)))
        pad.append((max(-start, 0), max(stop - n, 0)))
    return np.pad(x[tuple(crop)], pad)  # pads with 0 == the clipped minimum


def preprocess(image_path, out_path, mask_path=None, device="cuda"):
    img = nib.load(image_path)
    mask = nib.load(mask_path) if mask_path else hdbet_mask(image_path, device)

    # Resample axes finer than MIN_RES_MM; cubic for the image, nearest for the mask.
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    dxyz = [max(round(z, 2), MIN_RES_MM) / z for z in zooms]
    img = to_lps(resample(img, dxyz, order=3))
    mask = to_lps(resample(mask, dxyz, order=0))

    # Robust intensity clipping and uint16 quantization.
    x = img.get_fdata(dtype=np.float32)
    lo, hi = np.percentile(x, [P_LOW, P_HIGH])
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    x = np.rint(x * 65535).astype(np.uint16)

    # Convert the target FOV from mm to voxels, then crop/pad around the brain.
    # (Rounding guards against float error, e.g. 224 / 0.8 == 279.999...)
    spacing = [round(float(z), 2) for z in img.header.get_zooms()[:3]]
    target_shape = [math.floor(round(f / s, 6)) for f, s in zip(TARGET_FOV_MM, spacing)]
    x = crop_pad_to_brain(x, mask.get_fdata() > 0.5, target_shape)

    out = nib.Nifti1Image(x, img.affine)
    out.header.set_xyzt_units("mm")
    out.set_qform(img.affine, code=2)
    out.set_sform(img.affine, code=1)
    out.to_filename(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp-fpath", type=Path, required=True)
    parser.add_argument("--out-fpath", type=Path, required=True)
    parser.add_argument("--mask-fpath", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    preprocess(args.inp_fpath, args.out_fpath, args.mask_fpath, args.device)