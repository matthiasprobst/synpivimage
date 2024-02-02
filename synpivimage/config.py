import pathlib
import warnings

from pydantic import BaseModel
from typing import Union


class SynPivConfig(BaseModel):
    ny: int
    nx: int
    square_image: bool
    bit_depth: int
    noise_baseline: float
    dark_noise: float  # noise_baseline: 20, dark_noise: 2.29,
    shot_noise: bool
    sensitivity: float
    qe: float  # quantum efficiency. efficiency of photons to electron conversion
    particle_number: int
    particle_size_mean: float
    particle_size_std: float
    laser_width: int
    laser_shape_factor: int = 2
    image_particle_peak_count: int = 1000
    # laser_max_intensity: 1000
    particle_position_file: Union[str, pathlib.Path, None] = None
    particle_size_illumination_dependency: bool = True

    def __getitem__(self, item):
        warnings.warn(f'Please use .{item}', DeprecationWarning)
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f'invalid key: {item}')

    def __setitem__(self, key, value):
        warnings.warn(f'Please item assignment: {key}={value}', DeprecationWarning)
        setattr(self, key, value)


def get_default():
    return SynPivConfig(
        square_image=True,
        ny=128,
        nx=128,
        bit_depth=16,
        noise_baseline=100,
        dark_noise=4,
        shot_noise=True,
        sensitivity=0.5,
        qe=0.25,
        particle_number=1,
        particle_size_mean=2.5,
        particle_size_std=0,
        laser_width=2.,
        laser_shape_factor=2,
        relative_laser_intensity=1.0
    )
