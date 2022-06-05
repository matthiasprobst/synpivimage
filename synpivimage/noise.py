import numpy as np


def add_gaussian_noise(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """adds gaussian noise to an array"""
    if mean == 0:
        return arr
    row, col = arr.shape
    gauss = np.random.normal(mean, std, (row, col))
    _img = arr + gauss
    _img[_img < 0] = 0
    return _img


def add_camera_noise(input_irrad_photons: np.ndarray,
                     qe: float = 1., sensitivity: float = 1.,
                     dark_noise: float = 2.29, baseline: float = 0.,
                     enable_shot_noise: bool = True, seed: int = None) -> np.ndarray:
    """
    Generates camera noise and adds it to input array. Array with camera noise is returned.
    Code is shamelessly taken from http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
    and slightly adjusted (especially shot noise!)


    Parameters
    ----------
    input_irrad_photons: arr
        incoming photons. If counts are known, then multiply your counts with 1/sensitivity
    qe: float
        Quantum efficiency. Dependent on wavelength. It is the conversion factor from
        photons to electrons.
    sensitivity: float
        Represents the amplification of the voltage in the pixel from the
        photoelectrons and is also a property of the camera. [ADU/e-]
    dark_noise: float
        std of gaussian distribution
    baseline: float
        mean of gaussian distribution
    enable_shot_noise: bool, default=True
        Enables shot noise. Default is True.
    seed: int, default is None
        Seed for random state class. Use a value for "reproducable radnomness"

    """
    rs = np.random.RandomState(seed=seed)
    # Add shot noise
    if enable_shot_noise:
        photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)
    else:
        photons = input_irrad_photons

    # Convert to electrons
    electrons = qe * photons

    # Add dark noise
    electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons

    # Convert to ADU and add baseline
    adu = electrons_out * sensitivity  # Convert to discrete numbers
    adu += baseline

    adu[adu < 0] = 0  # just to be sure

    return adu
