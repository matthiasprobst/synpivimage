import enum
import numpy as np
import scipy
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union, Dict

SQRT2 = np.sqrt(2)
PARTICLE_INFLUENCE_FACTOR = 6


class ParticleFlag(enum.Enum):
    """Particle status flags."""
    INACTIVE = 0
    # ACTIVE = 1  # ILLUMINATED (in laser sheet) and in FOV
    ILLUMINATED = 1  # ILLUMINATED (in laser sheet) and in FOV
    IN_FOV = 2  # not captured by the sensor because it is out of the field of view
    OUT_OF_PLANE = 4  # particle not in laser sheet in z-direction should be same as weakly illuminated
    DISABLED = 8
    ACTIVE = 2 + 1  # ILLUMINATED (in laser sheet) and in FOV
    # OUT_OF_PLANE = 4
    # EXITED_FOV = 8  # in x or y direction due to displacement
    # IN_FOV = 16  # not captured by the sensor because it is out of the field of view


@dataclass
class ParticleDisplacement:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    size: np.ndarray
    intensity: np.ndarray
    flagA: np.ndarray
    flagB: np.ndarray

    def __repr__(self):
        return f'ParticleDisplacement()'


class Particles:
    """Particle class

    Contains position, size and flag information:
    - pixel (!) position: (x,y,z) within the light sheet. Mid of light sheet is z=0.
    - size: (real)) particle size in arbitrary units.
    - flag: Indicating status of a particle (active, out of plane, ...)
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 size: np.ndarray,
                 intensity: np.ndarray = None,
                 flag: np.ndarray = None):
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
            self.size = np.ones_like(self.x) * size
        else:
            self.size = size

        if intensity is None:
            self.intensity = np.zeros_like(self.x)
        else:
            self.intensity = intensity

        if flag is None:
            self.flag = np.zeros_like(self.x, dtype=int)
        else:
            self.flag = flag

        assert self.x.size == self.y.size == self.z.size == self.size.size == self.intensity.size == self.flag.size

    def __len__(self):
        return self.x.size

    def __getitem__(self, item):
        return Particles(x=self.x[item],
                         y=self.y[item],
                         z=self.z[item],
                         size=self.size[item],
                         intensity=self.intensity[item],
                         flag=self.flag[item])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def dict(self) -> Dict:
        """Returns a dictionary representation of the particle data"""
        return {'x': self.x,
                'y': self.y,
                'z': self.z,
                'size': self.size,
                'intensity': self.intensity,
                'flag': self.flag}

    def model_dump(self) -> Dict:
        """Returns a dictionary representation of the particle data where list instead of
        numpy arrays are used. This is useful for dumping to JSON"""
        return {'x': self.x.tolist(),
                'y': self.y.tolist(),
                'z': self.z.tolist(),
                'size': self.size.tolist(),
                'intensity': self.intensity.tolist(),
                'flag': self.flag.tolist()}

    def displace(self, dx=None, dy=None, dz=None):
        """Displace the particles. Can only be done if particles are not inactive.

        Raises
        ------
        ValueError
            If particles are inactive, which means that the particles have not been illuminated yet, hence
            they cannot be displaced. Call `synpivimage.take_image` first
        """
        if self.inactive.sum() > 0:
            raise ValueError("Cannot displace particles if they have been illuminated once, so a image has been taken")

        if dx is not None:
            new_x = self.x + dx
        else:
            new_x = self.x
        if dy is not None:
            new_y = self.y + dy
        else:
            new_y = self.y
        if dz is not None:
            new_z = self.z + dz
        else:
            new_z = self.z
        return self.__class__(x=new_x,
                              y=new_y,
                              z=new_z,
                              size=self.size,
                              intensity=None,
                              flag=None)

    @property
    def inactive(self):
        """Return mask of inactive particles"""
        return np.asarray(self.flag & ParticleFlag.INACTIVE.value, dtype=bool)

    @property
    def active(self):
        """Return mask of illuminated particles"""
        flag = ParticleFlag.ACTIVE.value
        return np.asarray(self.flag & flag == flag, dtype=bool)

    @property
    def n_active(self):
        """Return the number of active particles"""
        return np.sum(self.active)

    @property
    def source_density_number(self):
        """Return the number of particles in the laser sheet (and FOV)"""
        return np.sum(self.active)

    @property
    def disabled(self):
        """Return mask of disabled particles"""
        return np.asarray(self.flag & ParticleFlag.DISABLED.value, dtype=bool)

    @property
    def in_fov(self):
        """Return mask of particles in the FOV"""
        flag = ParticleFlag.IN_FOV.value
        return np.asarray(self.flag & flag, dtype=bool)

    @property
    def out_of_plane(self):
        """Return mask of particles out of plane"""
        flag = ParticleFlag.OUT_OF_PLANE.value
        return np.asarray(self.flag & flag == flag, dtype=bool)

    @property
    def in_fov_and_out_of_plane(self):
        """Return mask of particles out of plane"""
        flag = ParticleFlag.IN_FOV.value | ParticleFlag.OUT_OF_PLANE.value
        return np.asarray(self.flag & flag == flag, dtype=bool)

    @property
    def n_out_of_plane_loss(self) -> int:
        """Return the number of particles that are out of plane"""
        return np.sum(self.in_fov_and_out_of_plane)

    def info(self):
        """Prints some useful information about the particles"""
        print("=== Particle Information === ")
        print(f" > Number of simulated particles: {self.x.size}")
        print(f" > Number of active (illuminated and in FOV) particles: {self.active.sum()}")
        flag = ParticleFlag.IN_FOV.value
        n_in_fov = np.sum(self.flag & flag == flag)
        print(f" > Number of particles outside of FOV: {self.x.size - n_in_fov}")
        print(f" > Out of plane particles: {self.n_out_of_plane_loss}")
        # print(f" > Disabled particles due to out-of-FOV: {self.out_of_fov.sum()}")

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
        flag = np.zeros_like(x, dtype=bool)  # disabled by default
        return cls(x, y, z, size, intensity, flag)

    def __sub__(self, other: "Particles") -> ParticleDisplacement:
        """Subtract two particle sets"""
        return ParticleDisplacement(x=self.x - other.x,
                                    y=self.y - other.y,
                                    z=self.z - other.z,
                                    size=self.size - other.size,
                                    intensity=self.intensity - other.intensity,
                                    flagA=self.flag,
                                    flagB=other.flag)

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
