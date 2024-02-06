import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow(img, ax=None, **kwargs):
    """plt.imshow for PIV images (imshow + colorbar)"""
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(img, cmap='gray', **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)

    return ax, cax, cb
