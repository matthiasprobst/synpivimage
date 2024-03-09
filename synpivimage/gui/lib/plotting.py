import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit


def gauss(x, x0, k, C):
    return C * np.exp((-(x0 - x) ** 2) / k)


def plot_subpeak(R3, peak_loc, ax=None, color=None):
    """R3 = R[-1], R[0], R[1]"""
    if ax is None:
        ax = plt.gca()
    popt, pcov = curve_fit(gauss, [-1, 0, 1], R3)
    ax.scatter([-1, 0, 1], R3, color=color)
    _x = np.linspace(-2, 2, 101)
    g = gauss(_x, *popt)
    ax.plot(_x, g, color=color)
    ax.scatter(peak_loc, g.max(), marker='x', color=color)
    return ax


class GaussFit:
    def __init__(self, x0, k, C):
        self.x0 = x0
        self.k = k
        self.C = C

    def __call__(self, x):
        return gauss(x, self.x0, self.k, self.C)


def gauss3ptfit(data) -> GaussFit:
    assert len(data) == 3
    popt, pcov = curve_fit(gauss, [-1, 0, 1], data)
    _x = np.linspace(-2, 2, 101)
    return GaussFit(*popt)


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
