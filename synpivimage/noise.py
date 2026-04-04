from typing import Optional, Tuple, Union

import numpy as np

RandomSource = Union[np.random.RandomState, np.random.Generator]


def _as_random_source(rs: Optional[RandomSource]) -> RandomSource:
    """Return a random source; defaults to NumPy's global generator."""
    if rs is None:
        return np.random.default_rng()
    return rs


def _normal(
    rs: RandomSource,
    mean: float,
    std: float,
    shape: Tuple[int, ...],
) -> np.ndarray:
    if isinstance(rs, np.random.RandomState):
        return rs.normal(mean, std, shape)
    return rs.normal(mean, std, shape)


def _poisson(rs: RandomSource, photons: np.ndarray) -> np.ndarray:
    if isinstance(rs, np.random.RandomState):
        return rs.poisson(photons, size=photons.shape)
    return rs.poisson(photons, size=photons.shape)


def add_noise(
    irrad_photons: np.ndarray,
    shot_noise: bool,
    baseline: float,
    dark_noise: float,
    qe: float,
    rs: Optional[RandomSource] = None,
) -> np.ndarray:
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
    rs : Optional[RandomSource]
        Random source for reproducibility
    """
    rs = _as_random_source(rs)

    if shot_noise:
        shot_noise = compute_shot_noise(irrad_photons, rs)
        # converting to electrons
        electrons = qe * shot_noise
    else:
        electrons = qe * irrad_photons

    electrons_out = electrons + compute_dark_noise(
        baseline,
        dark_noise,
        electrons.shape,
        rs=rs,
    )
    return electrons_out


def compute_dark_noise(
    mean: float,
    std: float,
    shape: Tuple[int, ...],
    rs: Optional[RandomSource] = None,
) -> np.ndarray:
    """Add Gaussian dark/read noise to an array."""
    rs = _as_random_source(rs)
    return _normal(rs, mean, std, shape)


def compute_shot_noise(
    photons: np.ndarray,
    rs: Optional[RandomSource] = None,
) -> np.ndarray:
    """Based on the input photons, compute the poisson (shot noise) and return the noise array"""
    rs = _as_random_source(rs)
    return _poisson(rs, photons)
