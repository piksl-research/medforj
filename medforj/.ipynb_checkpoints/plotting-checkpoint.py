import io

import matplotlib
import numpy as np

matplotlib.use("Agg")  # For plotting consistently
import matplotlib.pyplot as plt
from PIL import Image


def to_np(x):
    return x.squeeze().detach().cpu().numpy()


def lightbox(x):
    """
    Plot a simple lightbox of a 3D image volume, iterating over the specified axis.

    Only 15 slices will be rendered, so the slices are subsampled.

    Assumes the input volume is in RAI orientation
    """

    vmin = x.min()
    vmax = x.max() * 0.8

    ncols = 5
    nrows = 3
    fig_scale = 2

    # Create subplots.
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * fig_scale, nrows * fig_scale))

    for row in range(3):
        # Bring the slice axis to the front.
        vol = np.moveaxis(x, row, 0)

        st = int(np.around(0.2 * vol.shape[0]))
        en = int(np.around(0.8 * vol.shape[0]))
        slice_idxs = np.around(np.linspace(st, en, ncols)).astype(np.uint16)

        # Plot each slice.
        for i, col in zip(slice_idxs, range(ncols)):
            match row:
                case 0:
                    img_slice = np.rot90(vol[i, ...])
                case 1:
                    img_slice = np.rot90(vol[i, ...])
                case _:
                    img_slice = vol[i, ...].T
            axs[row, col].imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axs[row, col].axis("off")

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

    # # Save figure to a buffer
    # buffer = io.BytesIO()
    # plt.savefig(buffer, format="png", bbox_inches="tight")
    # plt.close(fig)  # Close the figure to free memory
    # buffer.seek(0)

    # image = Image.open(buffer)

    # return image
