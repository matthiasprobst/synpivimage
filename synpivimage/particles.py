import numpy as np
import scipy
from copy import deepcopy
from typing import Tuple, Union

SQRT2 = np.sqrt(2)
PARTICLE_INFLUENCE_FACTOR = 6


class Particles:
    """Particle class"""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 size: np.ndarray,
                 intensity: np.ndarray = None,
                 mask: np.ndarray = None):
        if isinstance(x, (int, float)):
            self.x = np.array([x])
        else:
            self.x = x
        if isinstance(y, (int, float)):
            self.y = np.array([y])
        else:
            self.y = y
        if isinstance(z, (int, float)):
            self.z = np.array([z])
        else:
            self.z = z
        if isinstance(size, (int, float)):
            self.size = np.array([size])
        else:
            self.size = size
        if intensity is None:
            self.intensity = np.zeros_like(x)
        else:
            self.intensity = intensity
        if mask is None:
            self.mask = np.zeros_like(x, dtype=bool)
        else:
            self.mask = mask

    def __len__(self):
        return self.x.size

    def __getitem__(self, item):
        return Particles(x=self.x[item],
                         y=self.y[item],
                         z=self.z[item],
                         size=self.size[item],
                         intensity=self.intensity[item],
                         mask=self.mask[item])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def generate_uniform(cls,
                         n_particles: int,
                         size: Union[float, Tuple[float, float]],
                         x_bounds: Tuple[float, float],
                         y_bounds: Tuple[float, float],
                         z_bounds: Tuple[float, float]):
        """Generate particles uniformly"""
        assert len(x_bounds) == 2
        assert len(y_bounds) == 2
        assert len(z_bounds) == 2
        assert x_bounds[1] > x_bounds[0]
        assert y_bounds[1] > y_bounds[0]
        assert z_bounds[1] >= z_bounds[0]
        x = np.random.uniform(x_bounds[0], x_bounds[1], n_particles)
        y = np.random.uniform(y_bounds[0], y_bounds[1], n_particles)
        z = np.random.uniform(z_bounds[0], z_bounds[1], n_particles)

        if isinstance(size, (float, int)):
            size = np.ones_like(x) * size
        elif isinstance(size, (list, tuple)):
            assert len(size) == 2
            # generate a normal distribution, which is cut at +/- 2 sigma
            size = np.random.normal(size[0], size[1], n_particles)
            # cut the tails
            min_size = max(0, size[0] - 2 * size[1])
            max_size = size[0] + 2 * size[1]
            size[size < min_size] = 0
            size[size > max_size] = max_size
        else:
            raise ValueError(f"Size {size} not supported")
        intensity = np.zeros_like(x)  # no intensity by default
        mask = np.zeros_like(x, dtype=bool)  # disabled by default
        return cls(x, y, z, size, intensity, mask)

    def copy(self):
        """Return a copy of this object"""
        return deepcopy(self)


def compute_intensity_distribution(
        x,
        y,
        xp,
        yp,
        dp,
        sigmax,
        sigmay,
        fill_ratio_x,
        fill_ratio_y):
    """Computes the sensor intensity based on the error function as used in SIG by Lecordier et al. (2003)"""
    frx05 = 0.5 * fill_ratio_x
    fry05 = 0.5 * fill_ratio_y
    dxp = x - xp
    dyp = y - yp

    erf1 = (scipy.special.erf((dxp + frx05) / (SQRT2 * sigmax)) - scipy.special.erf(
        (dxp - frx05) / (SQRT2 * sigmax)))
    erf2 = (scipy.special.erf((dyp + fry05) / (SQRT2 * sigmay)) - scipy.special.erf(
        (dyp - fry05) / (SQRT2 * sigmay)))
    intensity = np.pi / 2 * dp ** 2 * sigmax * sigmay * erf1 * erf2
    return intensity


def model_image_particles(
        particles: Particles,
        nx: int,
        ny: int,
        sigmax: float,
        sigmay: float,
        fill_ratio_x: float,
        fill_ratio_y: float,
):
    """Model the photons irradiated by the particles on the sensor."""
    image_shape = (ny, nx)
    irrad_photons = np.zeros(image_shape)
    xp = particles.x
    yp = particles.y
    particle_sizes = particles.size
    part_intensity = particles.intensity
    delta = int(PARTICLE_INFLUENCE_FACTOR * max(sigmax, sigmay))
    for x, y, p_size, pint in zip(xp, yp, particle_sizes, part_intensity):
        xint = int(x)
        yint = int(y)
        xmin = max(0, xint - delta)
        ymin = max(0, yint - delta)
        xmax = min(nx, xint + delta)
        ymax = min(ny, yint + delta)
        sub_img_shape = (ymax - ymin, xmax - xmin)
        px = x - xmin
        py = y - ymin
        xx, yy = np.meshgrid(range(sub_img_shape[1]), range(sub_img_shape[0]))
        Ip = compute_intensity_distribution(
            x=xx,
            y=yy,
            xp=px,
            yp=py,
            dp=p_size,
            sigmax=sigmax,
            sigmay=sigmay,
            fill_ratio_x=fill_ratio_x,
            fill_ratio_y=fill_ratio_y,
        )
        irrad_photons[ymin:ymax, xmin:xmax] += Ip * pint
    return irrad_photons
