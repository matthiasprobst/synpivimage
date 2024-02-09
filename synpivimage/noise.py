import numpy as np
from typing import Tuple

from .random import rs


def add_noise(irrad_photons, shot_noise, baseline, dark_noise, qe):
    if shot_noise:
        shot_noise = compute_shot_noise(irrad_photons)
        # converting to electrons
        electrons = qe * shot_noise
    else:
        electrons = qe * irrad_photons

    if dark_noise > 0:
        electrons_out = electrons + compute_dark_noise(baseline, dark_noise, electrons.shape)
    else:
        electrons_out = electrons
    return electrons_out


def compute_dark_noise(mean: float, std: float, shape: Tuple[int, int]) -> np.ndarray:
    """adds gaussian noise to an array"""
    if mean == 0:
        return np.zeros(shape=shape)
    row, col = shape
    gnoise = np.random.normal(mean, std, (row, col))
    gnoise[gnoise < 0] = 0
    return gnoise


def compute_shot_noise(photons: np.ndarray) -> np.ndarray:
    """Based on the input photons, compute the poisson (shot noise) and return the noise array"""
    return rs.poisson(photons, size=photons.shape)
