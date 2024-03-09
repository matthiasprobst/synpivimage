from typing import Tuple

import numpy as np

SEED = None

np.random.seed(seed=SEED)

rs = np.random.RandomState(seed=SEED)


def add_noise(irrad_photons, shot_noise, baseline, dark_noise, qe, rs: rs = np.random.RandomState):
    """
    Add noise to an array of photons

    Parameters
    ----------
    irrad_photons : np.ndarray
        Array of photons
    shot_noise : bool
        If True, add shot noise to the array
    baseline : float
        Baseline signal
    dark_noise : float
        Dark noise (Standard deviation of the dark noise)
    qe : float
        Quantum efficiency
    rs : np.random.RandomState
        Random state for reproducibility
    """
    if shot_noise:
        shot_noise = compute_shot_noise(irrad_photons, rs)
        # converting to electrons
        electrons = qe * shot_noise
    else:
        electrons = qe * irrad_photons

    electrons_out = electrons + compute_dark_noise(baseline, dark_noise, electrons.shape)
    return electrons_out


def compute_dark_noise(mean: float, std: float, shape: Tuple[int, int]) -> np.ndarray:
    """adds gaussian noise to an array"""
    # if mean == 0:
    #     return np.zeros(shape=shape)
    gnoise = np.random.normal(mean, std, shape)
    gnoise[gnoise < 0] = 0
    return gnoise


def compute_shot_noise(photons: np.ndarray, rs: np.random.RandomState) -> np.ndarray:
    """Based on the input photons, compute the poisson (shot noise) and return the noise array"""
    return rs.poisson(photons, size=photons.shape)
