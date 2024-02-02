"""Core module"""
import h5py
import itertools
import multiprocessing as mp
import numpy as np
import os
import pathlib
import random
import warnings
import xarray as xr
import yaml
from dataclasses import dataclass
from math import ceil
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from typing import Dict, List, Union, Tuple

from . import noise
from ._version import __version__
from .config import SynPivConfig, DEFAULT_CFG

# from .noise import add_camera_noise

__this_dir__ = pathlib.Path(__file__).parent

SNT_FILENAME = __this_dir__ / 'standard_name_translation.yaml'

try:
    import h5rdmtoolbox as h5tbx

    h5tbx_is_available = True
except ImportError as e:
    h5tbx_is_available = False

if h5tbx_is_available:
    h5tbx.use('h5py')

CPU_COUNT = mp.cpu_count()

default_yaml_file: str = 'default.yaml'

PMIN_ALLOWED: float = 0.1


# default config has no noise since it can be added afterwards, too


# DEFAULT_CFG = ConfigParser()
# DEFAULT_CFG.read_dict(
#     dictionary={
#         'ny': 128,
#         'nx': 128,
#         'square_image': True,
#         'bit_depth': 16,
#         'noise_baseline': 0.0,
#         'dark_noise': 0.0,  # 'noise_baseline': 20, 'dark_noise': 2.29,
#         'sensitivity': 1,
#         'qe': 1,
#         'shot_noise': False,
#         'particle_number': 1,
#         'particle_size_mean': 2.5,
#         'particle_size_std': 0.25,
#         'laser_width': 3,
#         'laser_shape_factor': 2,
#         'sensor_gain': 1.0,  # a particle hit by max laser intensity will show max count on the sensor
#         # 'laser_max_intensity': 1000
#         'particle_position_file': None,
#         'particle_size_illumination_dependency': True
#     }
# )

@dataclass
class ParticleInfo:
    """Dataclass holding particle position, size and intensity information"""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    size: np.ndarray
    intensity: Union[np.ndarray, None] = None

    def __post_init__(self):
        if isinstance(self.x, (int, float)):
            self.x = np.array([self.x])

        if isinstance(self.y, (int, float)):
            self.y = np.array([self.y])

        if isinstance(self.z, (int, float)):
            self.z = np.array([self.z])

        if isinstance(self.size, (int, float)):
            self.size = np.array([self.size, ])

        assert len(self.x) == len(self.y)
        assert len(self.x) == len(self.z)
        assert len(self.x) == len(self.size)
        if self.intensity is not None:
            assert len(self.x) == len(self.intensity)

    @classmethod
    def from_hdf(cls, hdf_filename) -> List["ParticleInfo"]:
        """Init the class based on a HDF5 file. Expecting the particle infos to be located in
        group 'particle_infos'"""
        part_infos = []
        if isinstance(hdf_filename, list):
            for _hdf_filename in hdf_filename:
                part_infos.extend(cls.from_hdf(_hdf_filename))
            return part_infos

        with h5tbx.File(hdf_filename) as h5:
            grp_part_info = h5['particle_infos']
            for k, v in grp_part_info.items():
                x = v['x'][()]
                y = v['y'][()]
                z = v['z'][()]
                size = v['size'][()]
                intensity = v['intensity'][()]
                part_infos.append(cls(x=x, y=y, z=z, size=size, intensity=intensity))

        return part_infos

    def displace(self, dx=None, dy=None, dz=None):
        """Displace the particles"""
        if dx is not None:
            if isinstance(dx, (int, float)):
                self.x += dx
            else:
                self.x += np.array(dx)
        if dy is not None:
            if isinstance(dy, (int, float)):
                self.y += dy
            else:
                self.y += np.array(dy)
        if dz is not None:
            if isinstance(dz, (int, float)):
                self.z += dz
            else:
                self.z += np.array(dz)


def particle_location_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read particle location from file. Expected content (not checked) is
    comma separated with a header line
    #x, y, z, dp
     1, 0, 0, 2
     2, 0, 0, 2
     3, 1, 0, 2
    """
    posarr = np.genfromtxt(filename, delimiter=',')
    xp = posarr[:, 0]
    yp = posarr[:, 1]
    zp = posarr[:, 2]
    dp = posarr[:, 3]
    return xp, yp, zp, dp


def process_config_for_particle_position(cfg: SynPivConfig):
    """Process the config dictionary and return particle property (position + size)"""
    if not isinstance(cfg, SynPivConfig):
        raise TypeError(f'configuration must by SynPivConfig, not {type(cfg)}')
    if cfg.particle_position_file is None:
        return None
    xp, yp, zp, dp = particle_location_from_file(cfg['particle_position_file'])
    # update some config data:
    cfg.particle_number = len(xp)
    cfg.particle_size_mean = np.mean(dp)
    cfg.particle_size_std = np.std(dp)
    pposdict = dict(x=xp, y=yp, z=zp, size=dp)
    return ParticleInfo(**pposdict)


def write_yaml_file(filename: Union[str, bytes, os.PathLike], data: dict):
    """write data to a yaml file"""
    with open(filename, 'w') as f:
        yaml.dump(dict(data), f)
    return filename


def generate_default_yaml_file() -> pathlib.Path:
    """Writes the default configuration to the default yaml file"""
    write_yaml_file(default_yaml_file, DEFAULT_CFG)
    return pathlib.Path(default_yaml_file)


def yaml2dict(yaml_file: Union[str, bytes, os.PathLike]) -> Dict:
    """Converts a yaml file into a dictionary"""
    with open(yaml_file, 'r') as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def read_config(filename: Union[str, bytes, os.PathLike]) -> Dict:
    """Reads the yaml configuration from file and returns it as a dictionary"""
    from .config import SynPivConfig
    return SynPivConfig(**yaml2dict(filename))


def generate_image(
        config: SynPivConfig,
        particle_data: Union[ParticleInfo, None] = None,
        **kwargs) -> Tuple[np.ndarray, Dict, ParticleInfo]:
    """
    Generates a particle image based on a config (file). The generated image and the
    particle data as a dictionary is returned.
    Particle positions are generated randomly or may be set via particle_data
    (dictionary with keys x, y, z, s).
    In the latter case, the density
    set in the config is ignored
    """
    if isinstance(config, dict):
        config = SynPivConfig(**config)
    elif isinstance(config, SynPivConfig):
        warnings.warn('Please provide a SynPivConfig object',
                      DeprecationWarning)
        config = config
    else:
        warnings.warn('Please provide a SynPivConfig object',
                      DeprecationWarning)
        config = yaml2dict(config)

    # read and process configuration:
    if config['square_image']:
        config.ny = config.nx
    image_shape = (config.ny, config.nx)
    image_size = image_shape[0] * image_shape[1]

    sensitivity = config.sensitivity
    qe = config.qe
    dark_noise = config.dark_noise
    baseline = config.noise_baseline
    shot_noise_enabled = config.shot_noise

    bit_depth = config.bit_depth

    if particle_data is not None:
        xp = particle_data.x
        yp = particle_data.y
        zp = particle_data.z
        particle_sizes = particle_data.size
        if any(_arg is not None for _arg in (xp, yp, zp, particle_sizes)):
            if not all(_arg is not None for _arg in (xp, yp, zp, particle_sizes)):
                raise ValueError('If particle properties are set manually, all (xp, yp, zp and size)'
                                 'must be set!')
            if not isinstance(xp, np.ndarray):
                if isinstance(xp, (int, float)):
                    xp = np.array([xp])
                else:
                    xp = np.asarray(xp)
            if not isinstance(yp, np.ndarray):
                if isinstance(yp, (int, float)):
                    yp = np.array([yp])
                else:
                    yp = np.asarray(yp)
            if not isinstance(zp, np.ndarray):
                if isinstance(zp, (int, float)):
                    zp = np.array([zp])
                else:
                    zp = np.asarray(zp)
            if not isinstance(particle_sizes, np.ndarray):
                if isinstance(particle_sizes, (int, float)):
                    particle_sizes = np.array([particle_sizes])
                else:
                    particle_sizes = np.asarray(particle_sizes)
            # particle positions are set manually:
            for _arg in (xp, yp, zp, particle_sizes):
                if _arg.ndim != 1:
                    raise ValueError(f'particle information must be 1D and not {_arg.ndim}D: {_arg}')
            n_particles = xp.size
    else:
        if config.particle_number < 1:
            raise ValueError('Argument "particle_number" invalid. Must be an integer greater than 0: '
                             f'{config["particle_number"]}')
        n_particles = int(config.particle_number)
        assert n_particles > 0
        pmean = config.particle_size_mean  # mean particle size
        pstd = config.particle_size_std  # standard deviation of particle size
        pmin = pmean - 3 * pstd  # min particle size is 3*sigma below mean psize
        if pmin < PMIN_ALLOWED:
            warnings.warn(f'Particles smaller then {PMIN_ALLOWED} are set to {PMIN_ALLOWED}.')
            pmin = PMIN_ALLOWED
            if pmean <= pmin:
                raise ValueError('Mean particle size must be larger than smallest particle size!')
        pmax = pmean + 3 * pstd  # max particle size is 3*sigma above mean psize

    ppp = n_particles / image_size  # real ppp

    q = 2 ** bit_depth
    dz0 = config.laser_width
    s = config.laser_shape_factor

    # seed particles:
    irrad_photons = np.zeros(image_shape)
    if particle_data is None:
        xy_seeding_method = kwargs.pop('xy_seeding_method', 'random')
        zminmax = _compute_max_z_position_from_laser_properties(dz0, s)
        if xy_seeding_method == 'linear':
            _xp = np.linspace(0, image_shape[1], int(np.sqrt(n_particles)))
            _yp = np.linspace(0, image_shape[0], int(np.sqrt(n_particles)))
            xx, yy = np.meshgrid(_xp, _yp)
            xp = xx.ravel()
            yp = yy.ravel()
        else:
            xp = np.random.random(n_particles) * image_shape[1]
            yp = np.random.random(n_particles) * image_shape[0]
        zp = np.random.random(n_particles) * zminmax * 2 - zminmax  # physical location in laser sheet! TODO: units??!!
        # we should not clip the normal distribution
        # particle_sizes = np.clip(np.random.normal(pmean, pstd, n_particles), pmin, pmax)
        # but rather redo the normal distribution for the outliers:
        if pstd > 0:
            particle_sizes = np.random.normal(pmean, pstd, n_particles)
        else:
            particle_sizes = np.ones(n_particles) * pmean
        iout = np.argwhere((particle_sizes < pmin) | (particle_sizes > pmax))
        for i in iout[:, 0]:
            dp = np.random.normal(pmean, pstd)
            while dp < pmin or dp > pmax:
                dp = np.random.normal(pmean, pstd)
            particle_sizes[i] = dp

    # illuminate:
    if config.particle_size_illumination_dependency:
        part_intensity = particle_intensity(zp, dz0, s, dp=particle_sizes)
        part_intensity = part_intensity / (np.pi * max(particle_sizes) ** 2 / 8)
    else:
        part_intensity = particle_intensity(zp, dz0, s)

    # Computation of the particle intensity:
    # particle intensity = laser intensity x q x sensor gain
    # laser intensity is maximal 1
    # q is the bit depth

    # part_intensity [0, 1]
    # q --> bit depth, e.g. 8 or 16

    relative_laser_intensity = config.image_particle_peak_count / (
            2 ** config.bit_depth) / config.qe / config.sensitivity
    part_intensity = part_intensity * 2 ** bit_depth * relative_laser_intensity
    ny, nx = image_shape
    # nsigma = 4
    for x, y, psize, pint in zip(xp, yp, particle_sizes, part_intensity):
        delta = int(
            10 * psize)  # range plus minus the particle position, which changes the counts. theoretically all pixels must be changed, but it is only significant cose by
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
        squared_dist = (px - xx) ** 2 + (py - yy) ** 2
        irrad_photons[ymin:ymax, xmin:xmax] += np.exp(-8 * squared_dist / psize ** 2) * pint

    # so far, we computed the "photons" emitted from the particle --> num_photons

    # let's now compute the noise. First we compute the photon shot noise (https://en.wikipedia.org/wiki/Shot_noise):
    if shot_noise_enabled:
        shot_noise = noise.shot_noise(irrad_photons)
        # converting to electrons
        electrons = qe * shot_noise
    else:
        electrons = qe * irrad_photons

    if dark_noise > 0:
        electrons_out = electrons + noise.dark_noise(baseline, dark_noise, electrons.shape)
    else:
        electrons_out = electrons

    max_adu = int(2 ** bit_depth - 1)
    adu = electrons_out * sensitivity
    # if baseline > 0:
    #     adu += baseline

    # # add noise (pass the number of photons! That's why multiplication with sensitivity)
    # if dark_noise == 0. and not shot_noise_enabled and baseline == 0.:
    #     pass  # add NO noise
    # else:
    #     _intensity = add_camera_noise(_intensity / sensitivity, qe=qe, sensitivity=sensitivity,
    #                                   dark_noise=dark_noise, baseline=baseline,
    #                                   enable_shot_noise=shot_noise_enabled)

    # max_adu = int(2 ** bit_depth - 1)
    _saturated_pixels = adu > max_adu
    n_saturated_pixels = np.sum(_saturated_pixels)
    adu[adu > max_adu] = max_adu  # model saturation
    #
    # _intensity[_saturated_pixels] = max_adu  # models pixel saturation

    attrs = {'bit_depth': bit_depth,
             'noise_baseline': baseline,
             'noise_darknoise': dark_noise,
             'noise_eq': qe,
             'noise_sensitivity': sensitivity,
             'n_saturated_pixels': n_saturated_pixels,
             # 'ps_mean': np.mean(particle_sizes),
             # 'ps_std': np.std(particle_sizes),
             'ppp': ppp,
             'n_particles': n_particles,
             'laser_width': dz0,
             'laser_shape_factor': s,
             'laser_max_intensity': q,
             'particle_size_mean': config.particle_size_mean,
             'particle_size_std': config.particle_size_std,
             'image_particle_peak_count': config.image_particle_peak_count,
             'code_source': 'https://git.scc.kit.edu/da4323/piv-particle-density',
             'version': __version__}

    particle_info = ParticleInfo(**{'x': xp, 'y': yp, 'z': zp,
                                    'size': particle_sizes,
                                    'intensity': part_intensity})
    if bit_depth == 8:
        return adu.astype(np.uint8), attrs, particle_info
    elif bit_depth == 16:
        return adu.astype(np.uint16), attrs, particle_info
    return adu.astype(int), attrs, particle_info


def _compute_max_z_position_from_laser_properties(dz0: float, s: int) -> float:
    """If the particle intensity is below the noise mean value, it should not be
    simulated. Therefore, a min/max z value can be computed from the noise level and the laser
    properties. For a laser beam shape of 0 this laser intensity is constant for all z, thus
    an arbitrary value (1) is returned"""
    if s == 0:
        return 1
    return dz0 * (2 ** (0.5 - s)) ** (1 / (2 * s)) * np.pi ** (1 / (4 * s)) * (1 / np.sqrt(2 * np.pi) -
                                                                               np.log(np.sqrt(np.exp(1)) ** (
                                                                                       1 / np.sqrt(
                                                                                   2 * np.pi)) - 1)) ** (
                   1 / (2 * s))


def particle_intensity(z: np.ndarray, dz0: float, s: int, dp: np.ndarray = None):
    """Intensity of a particle in the laser beam.
    Max intensity of laser is 1 in this function.
    In previous versions this could be set. Now, the user has to take care of it him/herself

    Parameters
    ----------
    z : array-like
        particle z-position
    dz0 : float
        laser beam width. for 2 profile is gaussian
    s : int
        shape factor
    dp: array-like
        image particle diameter
    """
    if s == 0:
        if dp is None:
            return np.ones_like(z)
        return np.ones_like(z) * dp ** 2 * np.pi / 8
    if dp is None:
        return np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s)
    return np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s) * dp ** 2 * np.pi / 8


def _generate(
        cfgs: List[SynPivConfig],
        nproc: int,
        particle_information: Union[ParticleInfo, None]
) -> Tuple[np.ndarray, List[Dict], List[ParticleInfo]]:
    """Generates the particle image(s) and returns those alongside with the particle
    information hidden in the image(s)

    Parameters
    ----------
    cfgs: List[Dict]
        List of configuration to be passed to `generate_image`
    nproc: int, default=CPU_COUNT
        Number of processors to be used to generate the data
    particle_information: Union[List[Dict], Dict, None]
        List of dictionaries with the particle information. If None, the particle information
        will be generated randomly withing the configuration range

    Returns
    -------
    intensities: np.ndarray
        PIV image of size (n, ny, nx) where n=number of images and image size (ny, nx)
    particle_information: List[Dict]
        List of dictionary with the information like particle position, size, ... for the
        generated images
    """
    if isinstance(cfgs, SynPivConfig):
        cfgs = [cfgs, ]

    if nproc > CPU_COUNT:
        warnings.warn('The number of processors you provided is larger than the '
                      f'maximum of your computer. Will continue with {CPU_COUNT} processors instead.')
        _nproc = CPU_COUNT
    else:
        _nproc = nproc
    if cfgs[0].square_image:
        cfgs[0].ny = cfgs[0].nx
    intensities = np.empty(shape=(len(cfgs), cfgs[0].ny, cfgs[0].nx))
    # intensities = xr.DataArray(name='intensity', dims=('y', 'x'),
    #                            data=np.empty(shape=(len(cfgs), cfgs[0].ny, cfgs[0].nx)))

    generate_particle_positions = particle_information is None
    particle_information_out = []

    if particle_information is not None:
        assert isinstance(particle_information, (list, ParticleInfo))
    if isinstance(particle_information, list):
        assert len(particle_information) == len(cfgs)

    attrs_ls = []
    if _nproc < 2:
        for idx, _cfg in tqdm(enumerate(cfgs), total=len(cfgs), unit='cfg dict'):
            if generate_particle_positions:
                particle_data = process_config_for_particle_position(_cfg)
            else:
                if isinstance(particle_information, list):
                    particle_data = particle_information[idx]
                else:
                    particle_data = particle_information

            _intensity, _attrs, _partpos = generate_image(_cfg, particle_data=particle_data)
            intensities[idx, ...] = _intensity
            attrs_ls.append(_attrs)
            particle_information_out.append(_partpos)
        return intensities, attrs_ls, particle_information_out
    else:
        with mp.Pool(processes=_nproc) as pool:
            if generate_particle_positions:
                results = [pool.apply_async(
                    generate_image,
                    args=(_cfg, process_config_for_particle_position(_cfg))) for
                    _cfg in cfgs]
            else:
                if isinstance(particle_information, list):
                    results = [pool.apply_async(generate_image, args=(_cfg, particle_information[idx])) for
                               idx, _cfg in enumerate(cfgs)]
                else:
                    results = [pool.apply_async(generate_image, args=(_cfg, particle_information)) for
                               _cfg in cfgs]

            for i, r in tqdm(enumerate(results), total=len(results)):
                intensity, _attrs, particle_meta = r.get()
                intensities[i, ...] = intensity
                attrs_ls.append(_attrs)
                particle_information_out.append(particle_meta)
            return intensities, attrs_ls, particle_information_out


def apply_standard_names(h5: h5py.Group, standard_name_translation_filename):
    standard_name_translation_filename = pathlib.Path(standard_name_translation_filename)
    if not standard_name_translation_filename.exists():
        raise FileExistsError(f'File not found: {standard_name_translation_filename}')

    with open(standard_name_translation_filename, 'r') as f:
        snt_translation_dict = yaml.safe_load(f)

    for name, ds in h5.items():
        if isinstance(ds, h5py.Dataset):
            if name.strip('/') in snt_translation_dict:
                ds.attrs['standard_name'] = snt_translation_dict[name.strip('/')]


@dataclass
class ConfigManager:
    """Configuration class which manages creation of images and labels from one or multiple configurations"""
    cfgs: Tuple[Dict]

    def __post_init__(self):
        if isinstance(self.cfgs, Dict):
            self.cfgs = (self.cfgs,)

    def __repr__(self):
        return f'<ConfigManager ({len(self.cfgs)} configurations)>'

    def __len__(self):
        return len(self.cfgs)

    @staticmethod
    def from_variation_dict(initial_cfg: Dict,
                            variation_dict,
                            per_combination: int,
                            shuffle: bool):
        """inits the class based on the individual ranges of variables defined in the variation_dict"""
        return build_ConfigManager(initial_cfg=initial_cfg,
                                   variation_dict=variation_dict,
                                   per_combination=per_combination,
                                   shuffle=shuffle)

    def generate(self,
                 data_directory: Union[str, bytes, os.PathLike],
                 particle_info: Union[ParticleInfo, List[ParticleInfo], None] = None,
                 prefix='ds',
                 suffix='.hdf',
                 create_labels: bool = True,
                 overwrite: bool = False,
                 nproc: int = CPU_COUNT,
                 compression: str = 'gzip',
                 compression_opts: int = 5,
                 n_split: int = 10000) -> List[pathlib.Path]:
        """returns the generated data (intensities and particle information)
        This will not return all particle image information. Only number of particles!"""
        return self._generate_and_store_in_hdf(data_directory=data_directory,
                                               particle_info=particle_info,
                                               prefix=prefix,
                                               suffix=suffix,
                                               create_labels=create_labels,
                                               overwrite=overwrite,
                                               nproc=nproc,
                                               compression=compression,
                                               compression_opts=compression_opts,
                                               n_split=n_split)

    def to_hdf(self, *args, **kwargs) -> List[pathlib.Path]:
        """deprecated method --> generate()"""
        warnings.warn('The method "to_hdf" is deprecated. Use "generate" instead', DeprecationWarning)
        return self.generate(*args, **kwargs)

    def _generate_and_store_in_hdf(self,
                                   *,
                                   data_directory: Union[str, bytes, os.PathLike],
                                   prefix='ds',
                                   suffix='.hdf',
                                   particle_info: Union[List[ParticleInfo], ParticleInfo, None] = None,
                                   create_labels: bool = True,
                                   overwrite: bool = False,
                                   nproc: int = CPU_COUNT,
                                   compression: str = 'gzip',
                                   compression_opts: int = 5,
                                   n_split: int = 10000) -> List[pathlib.Path]:
        """
        Generates the images and writes data in chunks to multiple files according to chunking.
        Besides, the generated image, the following meta information are also stored with the intention
        that they will be used as labels:
        - number of particles
        - particle size mean
        - particle size std
        - intensity mean
        - intensity std

        Files are stored in data_directory and are named <prefix>XXXXXX<suffix> where XXXXXX is the index of
        the file.

        Parameters
        ----------
        data_directory: str, bytes, os.PathLike
            Path to directory where HDF5 files should be stored.
        create_labels: bool, default=True
            Whether to create label dataset or not.
        overwrite: bool, default=False
            Whether to overwrite existing data.
        nproc: int, default=CPU_COUNT
            Number of cores to use. Default uses maximum available.
        compression: str, default='gzip'
            Compression method to use when writing HDF5 data
        compression_opts: int, default='5'
            Compression option to use when writing HDF5 data
        n_split: int
            Number of images per HDF5 file

        Returns
        -------
        List of filenames
        """
        _dir = pathlib.Path(data_directory)
        _ds_filenames = list(_dir.glob(f'{prefix}*{suffix}'))

        if len(_ds_filenames) > 0 and not overwrite:
            raise FileExistsError(f'File exists and overwrite is False')
        elif len(_ds_filenames) > 0 and overwrite:
            for _ds_f in _ds_filenames:
                _ds_f.unlink()

        if not _dir.exists():
            _dir.mkdir(parents=True)

        def _chunk(lst: List, _n: int):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), _n):
                yield lst[i:i + _n]

        if n_split is None:
            _nfiles = 1
            chunked_cfgs = [self.cfgs, ]
        else:
            chunked_cfgs = _chunk(self.cfgs, n_split)
            _nfiles = ceil(len(self.cfgs) / n_split)
        print(f'Writing {len(self.cfgs)} dataset into {_nfiles} HDF5 file(s). This may take a while...')

        filenames = []
        for i_chunk, cfg_chunk in enumerate(chunked_cfgs):
            images, attrs, particle_information = _generate(cfg_chunk, nproc, particle_info)
            assert images.shape[0] == len(particle_information)
            assert images.shape[0] == len(attrs)
            new_name = f'{prefix}_{i_chunk:06d}{suffix}'
            new_filename = _dir.joinpath(new_name)
            filenames.append(new_filename)
            n_ds, ny, nx = images.shape
            with h5py.File(new_filename, 'w') as h5:
                ds_imageindex = h5.create_dataset('image_index', data=np.arange(0, n_ds, 1), dtype=int)
                ds_imageindex.attrs['long_name'] = 'image index'
                ds_imageindex.attrs['units'] = ''
                ds_imageindex.make_scale()

                ds_x_pixel_coord = h5.create_dataset('ix', data=np.arange(0, nx, 1), dtype=int)
                ds_x_pixel_coord.attrs['units'] = 'pixel'
                ds_x_pixel_coord.make_scale()

                ds_y_pixel_coord = h5.create_dataset('iy', data=np.arange(0, ny, 1), dtype=int)
                ds_y_pixel_coord.attrs['units'] = 'pixel'
                ds_y_pixel_coord.make_scale()

                ds_images = h5.create_dataset('images', shape=images.shape, compression=compression,
                                              compression_opts=compression_opts,
                                              chunks=(1, *images.shape[1:]))
                ds_images.attrs['long_name'] = 'image intensity'
                ds_images.attrs['units'] = 'count'

                if create_labels:
                    ds_labels = h5.create_dataset('labels', shape=images.shape, compression=compression,
                                                  compression_opts=compression_opts,
                                                  chunks=(1, *images.shape[1:]))
                    ds_labels.attrs['long_name'] = 'image label'
                    ds_labels.attrs['units'] = ' '

                ds_nparticles = h5.create_dataset('nparticles', shape=n_ds,
                                                  compression=compression,
                                                  compression_opts=compression_opts, dtype=int)
                ds_nparticles.attrs['long_name'] = 'number of particles'
                ds_nparticles.attrs['units'] = ''
                ds_nparticles.make_scale()

                ds_particle_density = h5.create_dataset('particle_density', shape=n_ds,
                                                        compression=compression,
                                                        compression_opts=compression_opts, dtype=float)
                ds_particle_density.attrs['long_name'] = 'particle density'
                ds_particle_density.attrs['units'] = '1/pixel'
                ds_particle_density.make_scale()

                ds_mean_size = h5.create_dataset('particle_size_mean', shape=n_ds, compression=compression,
                                                 compression_opts=compression_opts)
                ds_mean_size.attrs['units'] = 'pixel'
                ds_mean_size.make_scale()

                ds_configured_mean_size = h5.create_dataset('configured_particle_size_mean', shape=n_ds,
                                                            compression=compression,
                                                            compression_opts=compression_opts)
                ds_configured_mean_size.attrs['units'] = 'pixel'
                ds_configured_mean_size.make_scale()

                ds_std_size = h5.create_dataset('particle_size_std', shape=n_ds, compression=compression,
                                                compression_opts=compression_opts)
                ds_std_size.attrs['units'] = 'pixel'
                ds_std_size.make_scale()

                ds_configured_std_size = h5.create_dataset('configured_particle_size_std', shape=n_ds,
                                                           compression=compression,
                                                           compression_opts=compression_opts)
                ds_configured_std_size.attrs['units'] = 'pixel'
                ds_configured_std_size.make_scale()

                # ds_intensity_mean = h5.create_dataset('particle_intensity_mean', shape=n_ds, compression=compression,
                #                                       compression_opts=compression_opts)
                # ds_intensity_mean.attrs['units'] = 'count'
                # ds_intensity_mean.make_scale()
                #
                # ds_intensity_std = h5.create_dataset('particle_intensity_std', shape=n_ds, compression=compression,
                #                                      compression_opts=compression_opts)
                # ds_intensity_std.attrs['units'] = 'count'
                # ds_intensity_std.make_scale()

                ds_n_satpx = h5.create_dataset('number_of_saturated_pixels', shape=n_ds, compression=compression,
                                               compression_opts=compression_opts)
                ds_n_satpx.attrs['units'] = ''
                ds_n_satpx.make_scale()

                ds_laser_width = h5.create_dataset('laser_width', shape=n_ds, compression=compression,
                                                   compression_opts=compression_opts)
                ds_laser_width.attrs['units'] = 'm'
                ds_laser_width.make_scale()

                ds_bitdepth = h5.create_dataset('bit_depth', shape=n_ds, compression=compression,
                                                compression_opts=compression_opts, dtype=int)
                ds_bitdepth.attrs['units'] = 'count'
                ds_bitdepth.make_scale()

                ds_laser_shape_factor = h5.create_dataset('laser_shape_factor', shape=n_ds, compression=compression,
                                                          compression_opts=compression_opts)
                ds_laser_shape_factor.attrs['units'] = ''
                ds_laser_shape_factor.make_scale()

                ds_n_satpx[:] = [a['n_saturated_pixels'] for a in attrs]
                ds_laser_shape_factor[:] = [a['laser_shape_factor'] for a in attrs]
                ds_laser_width[:] = [a['laser_width'] for a in attrs]

                ds_images[:] = images
                ds_labels[:] = np.stack([generate_label(p_info.x,
                                                        p_info.y,
                                                        images.shape[1:],
                                                        False) for p_info in particle_information])
                assert ds_labels.shape == images.shape
                npart = np.asarray([len(p.x) for p in particle_information])
                ds_nparticles[:] = npart
                ds_particle_density[:] = npart / (nx * ny)
                ds_mean_size[:] = [np.mean(p.size) for p in particle_information]
                ds_configured_mean_size[:] = [a['particle_size_mean'] for a in attrs]
                ds_configured_std_size[:] = [a['particle_size_std'] for a in attrs]
                ds_std_size[:] = [np.std(p.size) for p in particle_information]
                # ds_intensity_mean[:] = [np.mean(p['intensity']) for p in particle_information]
                # ds_intensity_std[:] = [np.std(p['intensity']) for p in particle_information]
                ds_bitdepth[:] = [a['bit_depth'] for a in attrs]

                part_pos_grp = h5.create_group('particle_infos')
                for ipart, part_info in enumerate(particle_information):
                    grp = part_pos_grp.create_group(f'image_{ipart:06d}')
                    grp.create_dataset('x', data=part_info.x)
                    grp.create_dataset('y', data=part_info.y)
                    grp.create_dataset('z', data=part_info.z)
                    grp.create_dataset('size', data=part_info.size)
                    grp.create_dataset('intensity', data=part_info.intensity)

                for ds in (ds_imageindex, ds_nparticles, ds_mean_size, ds_std_size,
                           # ds_intensity_mean, ds_intensity_std,
                           ds_laser_width, ds_laser_shape_factor, ds_n_satpx,
                           ds_particle_density, ds_bitdepth,
                           ds_configured_mean_size,
                           ds_configured_std_size):
                    ds_images.dims[0].attach_scale(ds)
                    ds_labels.dims[0].attach_scale(ds)
                ds_images.dims[1].attach_scale(ds_y_pixel_coord)
                ds_images.dims[2].attach_scale(ds_x_pixel_coord)
                ds_labels.dims[1].attach_scale(ds_y_pixel_coord)
                ds_labels.dims[2].attach_scale(ds_x_pixel_coord)

                if h5tbx_is_available:
                    print('Processing standard names...')
                    apply_standard_names(h5, SNT_FILENAME)
                print('... done.')
        return filenames


def build_ConfigManager(*,
                        initial_cfg: Dict,
                        variation_dict: Dict,
                        per_combination: int = 1,
                        shuffle: bool = True) -> ConfigManager:
    """Generates a list of configuration dictionaries.
    Request an initial configuration and a tuple of variable length containing
    the name of a dictionary key of the configuration and the values to be chosen.
    A list containing configuration dictionaries of all combinations is
    returned. Moreover, a filename is generated and added to the dictionary.

    Parameters
    ----------
    initial_cfg : Dict
        Initial configuration to take and replace parameters to vary in
    variation_dict: Dict
        Dictionary defining the variable to be varied. The keyword is the variable name,
        the value must be a list/array-like, e.g. {'particle_number': [1, 2, 3]}
    per_combination: int=1
        Number of configurations per parameter set (It may be useful to repeat
        the generation of a specific parameter set because particles are randomly
        generated). Default is 1.
    shuffle: bool=True
        Shuffle the config files. Default is True.

    """

    # if variation has a float entry, make it a list:
    for k, v in variation_dict.items():
        if isinstance(v, (int, float)):
            variation_dict[k] = [v, ]

    keys, values = zip(*variation_dict.items())
    _dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    cfgs = []
    count = 0
    for _, param_dict in enumerate(_dicts):
        for _ in range(per_combination):
            cfgs.append(initial_cfg.copy())
            for k, v in param_dict.items():
                cfgs[-1][k] = v
                # a raw filename without extension to the dictionary:
            # cfgs[-1]['fname'] = f'ds{count:06d}'
            count += 1
    if shuffle:
        random.shuffle(cfgs)
    return ConfigManager(cfgs)


def generate_label(x, y, image_shape, ret_xr=True):
    """generate label based on x and y positions"""
    label = np.zeros(image_shape, dtype=np.float32)

    # loop over objects positions and marked them with 100 on a label
    # note: *_ because some datasets contain more info except x, y coordinates
    for _x, _y in zip(x, y):
        if _y <= image_shape[0] and _x <= image_shape[1]:
            label[int(_y)][int(_x)] += 100

    # apply a convolution with a Gaussian kernel
    label = gaussian_filter(label, sigma=(1, 1), order=0)

    if ret_xr:
        return xr.DataArray(dims=('y', 'x'), data=label, attrs={'long_name': 'Label for density map CNN',
                                                                'comment': 'Each particle got value 100 at '
                                                                           'the respective integer position. '
                                                                           'Afterwards, gaussian filter was applied. '
                                                                           'Sum of array divided by 100 will '
                                                                           'result in number of particles '
                                                                           'in the image'})
    return label


if __name__ == '__main__':
    pass
