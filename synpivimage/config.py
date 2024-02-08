import warnings

from pydantic import BaseModel


class SynPivConfig(BaseModel):
    ny: int
    nx: int
    bit_depth: int
    noise_baseline: float
    dark_noise: float  # noise_baseline: 20, dark_noise: 2.29,
    shot_noise: bool
    sensitivity: float
    qe: float  # quantum efficiency. efficiency of photons to electron conversion
    particle_number: int
    particle_size_mean: float
    particle_size_std: float
    laser_width: float
    laser_shape_factor: int = 2
    image_particle_peak_count: int = 1000
    fill_ratio_x: float = 1.0
    fill_ratio_y: float = 1.0
    # pattern_meanx = 2.,  # the width of the gaussian particle (constant for image, see SIG)
    # pattern_meany = 2.,  # the width of the gaussian particle (constant for image, see SIG)
    particle_size_definition: str = 'e2',  # other: 'I2', '2sigma'

    def __getitem__(self, item):
        warnings.warn(f'Please use .{item}', DeprecationWarning)
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f'invalid key: {item}')

    def __setitem__(self, key, value):
        warnings.warn(f'Please item assignment: {key}={value}', DeprecationWarning)
        setattr(self, key, value)

    def particle_density(self) -> float:
        """Return particle image density (projection of all particles in laser onto image plane).
        This value is also referred to as ppp (particle per pixel)"""
        return self.particle_number / (self.nx * self.ny)


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
        particle_size_mean=2.5,  # mean particle size of gaussian distribution for particle sizes
        particle_size_std=0.1,  # sigma of gaussian distribution for particle sizes
        pattern_meanx=2.,  # the width of the gaussian particle (constant for image, see SIG)
        pattern_meany=2.,  # the width of the gaussian particle (constant for image, see SIG)
        laser_width=2.,
        laser_shape_factor=2,
        relative_laser_intensity=1.0
    )
