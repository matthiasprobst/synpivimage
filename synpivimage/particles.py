import enum
import logging
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple
from typing import Union, Dict, Optional, List

import numpy as np
import scipy

from .component import Component

SQRT2 = np.sqrt(2)
PARTICLE_INFLUENCE_FACTOR = 6
logger = logging.getLogger('synpivimage')


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


class Particles(Component):
    """Particle class

    Contains position, size and flag information:
    - pixel (!) position: (x,y,z) within the light sheet. Mid of light sheet is z=0.
    - size: (real)) particle size in arbitrary units.
    - flag: Indicating status of a particle (active, out of plane, ...)
    """

    def __init__(self,
                 *,
                 x: Union[float, List[float], np.ndarray],
                 y: Union[float, List[float], np.ndarray],
                 z: Union[float, List[float], np.ndarray],
                 size: Union[float, List[float], np.ndarray],
                 source_intensity: Optional[Union[float, List[float], np.ndarray]] = None,
                 max_image_photons: Optional[Union[float, List[float], np.ndarray]] = None,
                 image_electrons: Optional[Union[float, List[float], np.ndarray]] = None,
                 image_quantized_electrons: Optional[Union[float, List[float], np.ndarray]] = None,
                 flag: Union[int, List[int], np.ndarray] = None):
        """
        Parameters
        ----------
        x : np.ndarray
            x-coordinate of the particles on the sensor in pixels
        y : np.ndarray
            y-coordinate of the particles on the sensor in pixels
        z : np.ndarray
            z-coordinate of the particles on the sensor in arbitrary units
        size : np.ndarray
            Particle size in pixels
        source_intensity : np.ndarray
            Source intensity of the particles, which is the intensity it emits as a point source.
            The peak intensity on the sensor may be higher. see property `irrad_photons`
        max_image_photons : np.ndarray
            Maximum number of photons on the sensor
        image_electrons : np.ndarray
            Number of electrons on the sensor
        image_quantized_electrons : np.ndarray
            Number of quantized electrons on the sensor
        """

        def _parse_array(_arr, _n: int, dtype=None):
            if _arr is None:
                _arr = np.zeros(shape=(_n,), dtype=dtype)
            if isinstance(_arr, (tuple, list)):
                _arr = np.asarray(_arr)
            elif isinstance(_arr, (float, int)):
                _arr = np.array([_arr, ])
            else:
                if not isinstance(_arr, np.ndarray):
                    raise TypeError(f"Expected array, got {type(_arr)}")
            if not _arr.ndim == 1:
                raise ValueError(f"Expected 1D array, got {_arr.ndim}D")
            if _n is None:
                return _arr
            if _arr.size != _n:
                raise ValueError(f"Expected array of size {_n}, got {_arr.size}")
            return _arr

        if x is None:
            raise ValueError("x cannot be None")
        self._x = _parse_array(x, None)
        N = self._x.size
        self._y = _parse_array(y, N)
        self._z = _parse_array(z, N)
        self._size = _parse_array(size, N)
        self._source_intensity = _parse_array(source_intensity, N)
        self._max_image_photons = _parse_array(max_image_photons, N)
        self._image_electrons = _parse_array(image_electrons, N)
        self._image_quantized_electrons = _parse_array(image_quantized_electrons, N)
        self._flag = _parse_array(flag, N, dtype=int)
        self._xlim = None
        self._ylim = None
        self._zlim = None

    @property
    def x(self):
        """x-coordinate of the particles on the sensor in pixels"""
        return self._x

    @property
    def y(self):
        """y-coordinate of the particles on the sensor in pixels"""
        return self._y

    @property
    def z(self):
        """z-coordinate of the particles on the sensor in arbitrary units"""
        return self._z

    @property
    def size(self):
        """Particle size in pixels"""
        return self._size

    @property
    def source_intensity(self):
        """Source intensity of the particles, which is the intensity it emits as a point source.
        The peak intensity on the sensor may be higher. see property `irrad_photons`"""
        return self._source_intensity

    @source_intensity.setter
    def source_intensity(self, value):
        assert value.ndim == 1, f"Expected 1D array, got {value.ndim}D"
        assert value.size == self.n_particles, f"Expected array of size {self.n_particles}, got {value.size}"
        self._source_intensity = value

    @property
    def max_image_photons(self):
        """Maximum number of photons on the sensor"""
        return self._max_image_photons

    @max_image_photons.setter
    def max_image_photons(self, value):
        assert value.ndim == 1, f"Expected 1D array, got {value.ndim}D"
        assert value.size == self.n_particles, f"Expected array of size {self.n_particles}, got {value.size}"
        self.max_image_photons = value

    @property
    def image_electrons(self):
        """Number of electrons on the sensor"""
        return self._image_electrons

    @image_electrons.setter
    def image_electrons(self, value):
        assert value.ndim == 1, f"Expected 1D array, got {value.ndim}D"
        assert value.size == self.n_particles, f"Expected array of size {self.n_particles}, got {value.size}"
        self.image_electrons = value

    @property
    def image_quantized_electrons(self):
        """Number of quantized electrons on the sensor"""
        return self._image_quantized_electrons

    @image_quantized_electrons.setter
    def image_quantized_electrons(self, value):
        assert value.ndim == 1, f"Expected 1D array, got {value.ndim}D"
        assert value.size == self.n_particles, f"Expected array of size {self.n_particles}, got {value.size}"
        self.image_quantized_electrons = value

    @property
    def flag(self):
        """Indicating status of a particle (active, out of plane, ...)"""
        return self._flag

    @property
    def irrad_photons(self):
        """The number of photons irradiated by the particles on the sensor"""
        return self._source_intensity

    @irrad_photons.setter
    def irrad_photons(self, value):
        assert value.ndim == 1, f"Expected 1D array, got {value.ndim}D"
        assert value.size == self.n_particles, f"Expected array of size {self.n_particles}, got {value.size}"
        self.irrad_photons = value

    @property
    def n_particles(self):
        """Return the number of particles. Equal to `len(self)`"""
        return self.x.size

    def reset(self):
        """Sets all intensities to zero and flags to zero"""
        self._source_intensity = np.zeros_like(self.x)
        self._max_image_photons = np.zeros_like(self.x)
        self._image_electrons = np.zeros_like(self.x)
        self._image_quantized_electrons = np.zeros_like(self.x)
        self._flag = np.zeros_like(self.x, dtype=int)

    @classmethod
    def generate(
            cls,
            ppp: float,
            dx_max: float,
            dy_max: float,
            dz_max: float,
            size: float,
            camera: "Camera",
            laser: "Laser"
    ) -> "Particles":
        """Generate particles based on a certain ppp (particles per pixel). With
        dx, dy, dz the maximum displacement of the particles can be set. The camera and laser
        are used to determine the sensor size and the laser width.

        Parameters
        ----------
        ppp : float
            Particles per pixel
        dx_max : float
            Maximum displacement in x-direction
        dy_max : float
            Maximum displacement in y-direction
        dz_max : float
            Maximum displacement in z-direction
        size : float
            Particle size
        camera : Camera
            Camera model
        laser : Laser
            Laser model
        """
        from .utils import generate_particles
        return generate_particles(ppp=ppp, dx_max=dx_max, dy_max=dy_max, dz_max=dz_max, size=size, camera=camera,
                                  laser=laser)

    def get_ppp(self, camera_size: int) -> float:
        """Return the particles per pixel"""
        return self.active.sum() / camera_size

    def regenerate(self) -> "Particles":
        """Regenerate the particles of this object.
        The locations (x, y, z) of the particles will change and the intensities
        will be reset

        .. note::

            Does NOT create a new object!
            The returned object is the same
        """
        self.reset()
        N = len(self.x)
        if self._xlim is None:
            self._x = np.random.uniform(min(self.x), max(self.x), N)
        else:
            self._x = np.random.uniform(*self._xlim, N)

        if self._ylim is None:
            self._y = np.random.uniform(min(self.y), max(self.y), N)
        else:
            self._y = np.random.uniform(*self._ylim, N)

        if self._ylim is None:
            self._z = np.random.uniform(min(self.z), max(self.z), N)
        else:
            self._z = np.random.uniform(*self._zlim, N)
        return self

    def __len__(self):
        return self.x.size

    def dict(self) -> Dict:
        """Returns a dictionary representation of the particle data"""
        return {'x': self.x,
                'y': self.y,
                'z': self.z,
                'size': self.size,
                'flag': self.flag,
                'source_intensity': self.source_intensity,
                'max_image_photons': self.max_image_photons,
                'image_electrons': self.image_electrons,
                'image_quantized_electrons': self.image_quantized_electrons}

    def __getitem__(self, item):
        data: Dict = self.dict()
        x = data['x'][item]
        if x.ndim == 0:
            return Particles(**{k: [v[item], ] for k, v in data.items()})
        return Particles(**{k: v[item] for k, v in data.items()})

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def model_dump(self) -> Dict:
        """Returns a dictionary representation of the particle data where list instead of
        numpy arrays are used. This is useful for dumping to JSON"""
        return {'x': self.x.tolist(),
                'y': self.y.tolist(),
                'z': self.z.tolist(),
                'size': self.size.tolist(),
                'source_intensity': self.source_intensity.tolist(),
                'max_image_photons': self.max_image_photons.tolist(),
                'image_electrons': self.image_electrons.tolist(),
                'image_quantized_electrons': self.image_quantized_electrons.tolist(),
                'flag': self.flag.tolist()}

    def save_jsonld(self, filename: Union[str, pathlib.Path]):
        from pivmetalib import pivmeta
        from pivmetalib import m4i
        from .codemeta import get_package_meta
        filename = pathlib.Path(filename)  # .with_suffix('.jsonld')

        source_intensity = 0. if np.all(self.source_intensity == 0) else self.source_intensity
        max_image_photons = 0. if np.all(self.max_image_photons == 0) else self.max_image_photons
        image_electrons = 0. if np.all(self.image_electrons == 0) else self.image_electrons
        image_quantized_electrons = 0. if np.all(
            self.image_quantized_electrons == 0) else self.image_quantized_electrons

        hasParameter = [
            m4i.variable.NumericalVariable(
                label='x',
                hasNumericalValue=self.x.astype("float16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='y',
                hasNumericalValue=self.y.astype("float16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='z',
                hasNumericalValue=self.z.astype("float16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='size',
                hasNumericalValue=self.size.astype("float16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='flag',
                hasNumericalValue=self.flag.astype("uint8").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='source_intensity',
                hasNumericalValue=np.asarray(source_intensity, dtype="uint16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='max_image_photons',
                hasNumericalValue=np.asarray(max_image_photons, dtype="uint16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='image_electrons',
                hasNumericalValue=np.asarray(image_electrons, dtype="uint16").tolist()
            ),
            m4i.variable.NumericalVariable(
                label='image_quantized_electrons',
                hasNumericalValue=np.asarray(image_quantized_electrons, dtype="uint16").tolist()
            ),
        ]
        particles = pivmeta.SyntheticParticle(
            hasSourceCode=get_package_meta(),
            hasParameter=hasParameter
        )
        with open(filename, 'w') as f:
            f.write(
                particles.model_dump_jsonld(
                    # context={
                    #     "@import": 'https://raw.githubusercontent.com/matthiasprobst/pivmeta/main/pivmeta_context.jsonld',
                    # }
                )
            )
        return filename

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
                              source_intensity=None,
                              max_image_photons=None,
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
                                    intensity=self.source_intensity - other.source_intensity,
                                    flagA=self.flag,
                                    flagB=other.flag)

    def copy(self):
        """Return a copy of this object"""
        return deepcopy(self)

    def load_jsonld(self):
        raise NotImplementedError("Not implemented yet")


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
    """Computes the sensor intensity based on the error function as used in SIG by Lecordier et al. (2003)

    Parameters
    ----------
    x : np.ndarray
        x-coordinate of the sensor pixels
    y : np.ndarray
        y-coordinate of the sensor pixels
    xp : float
        x-coordinate of the particles on the sensor in pixels
    yp : float
        y-coordinate of the particles on the sensor in pixels
    dp : float
        particle image diameter (in pixels)
    sigmax : float
        standard deviation of the Gaussian in x-direction
    sigmay : float
        standard deviation of the Gaussian in y-direction
    fill_ratio_x : float
        fill ratio of the sensor in x-direction
    fill_ratio_y : float
        fill ratio of the sensor in y-direction

    Returns
    -------
    np.ndarray
        The intensity distribution of the particles on the sensor

    """
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
    delta = int(PARTICLE_INFLUENCE_FACTOR * max(sigmax, sigmay))
    max_image_photons = np.zeros_like(particles.x)
    for ip, (x, y, p_size, pint) in enumerate(
            zip(particles.x, particles.y, particles.size, particles.source_intensity)):
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
        max_image_photons[ip] = np.max(Ip * pint)
    return irrad_photons, max_image_photons
