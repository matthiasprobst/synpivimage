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


def imshow2(img1, img2, axs=None, **kwargs):
    """Plots two images side by side with colorbars."""
    if axs is None:
        fig, axs = plt.subplots(1, 2, tight_layout=True)
    caxs = []
    cbs = []

    ax, cax, cb = imshow(img1, ax=axs[0], **kwargs)
    caxs.append(cax)
    cbs.append(cb)

    ax, cax, cb = imshow(img2, ax=axs[1], **kwargs)
    caxs.append(cax)
    cbs.append(cb)

    return axs, caxs, cbs
