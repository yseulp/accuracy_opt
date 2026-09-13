from scripts.model import indexing

import numpy as np
import matplotlib.pyplot as plt


def plot_xz(input, domain_dim, titletext, color_limits=None, slice_y=None):

    # re-format to 3D
    input = indexing.to_3d(input,domain_dim)
    
    # Use the configured slice, or preserve the previous central-slice default.
    if slice_y is None:
        slice_y = input.shape[1] // 2
    slice_img = np.transpose(input[:, slice_y, :])
    
    # Create the figure and axes
    fig, ax = plt.subplots()
    imshow_kwargs = {"cmap": "gray", "aspect": "auto"}
    if color_limits is not None:
        imshow_kwargs["vmin"], imshow_kwargs["vmax"] = color_limits
    im = ax.imshow(slice_img, **imshow_kwargs)
    fig.colorbar(im, ax=ax)
    
    ax.set_title(titletext)
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    return fig
    
