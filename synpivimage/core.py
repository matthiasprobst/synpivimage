import itertools
import multiprocessing as mp
import os
import pathlib
import random
import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Union, Tuple

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from ._version import __version__
from .noise import add_camera_noise

try:
    from h5rdmtoolbox.conventions import StandardNameTable, StandardNameTableTranslation
except ImportError as e:
    print(e)
np.random.seed()
CPU_COUNT = mp.cpu_count()

default_yaml_file = 'default.yaml'

PMIN_ALLOWED = 0.1

# default config has no noise since it can be added afterwards, too
DEFAULT_CFG = {'ny': 128, 'nx': 128, 'square_image': True,
               'bit_depth': 16,
               'noise_baseline': 0.0, 'dark_noise': 0.0,  # 'noise_baseline': 20, 'dark_noise': 2.29,
               'sensitivity': 1, 'qe': 1, 'shot_noise': False,
               'particle_number': 1, 'particle_size_mean': 2.5,
               'particle_size_std': 0.25,
               'laser_width': 3, 'laser_shape_factor': 2,
               'sensor_gain': 1.0,  # a particle hit by max laser intensity will show max count on the sensor
               # 'laser_max_intensity': 1000
               'particle_position_file': None,
               'particle_size_illumination_dependency': True
               }


def particle_location_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read particle loaction from file. Expected content (not checked) is
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


def process_config_for_particle_position(cfg: Dict):
    if cfg['particle_position_file'] is None:
        return None
    xp, yp, zp, dp = particle_location_from_file(cfg['particle_position_file'])
    # update some config data:
    cfg['particle_number'] = len(xp)
    cfg['particle_size_mean'] = np.mean(dp)
    cfg['particle_size_std'] = np.std(dp)
    pposdict = dict(x=xp, y=yp, z=zp, size=dp)
    return pposdict


def write_yaml_file(filename: Union[str, bytes, os.PathLike], data: dict):
    """write data to a yaml file"""
    with open(filename, 'w') as f:
        yaml.dump(data, f)
    return filename


def generate_default_yaml_file() -> None:
    """Writes the default configuration to the default yaml file"""
    write_yaml_file(default_yaml_file, DEFAULT_CFG)


def yaml2dict(yaml_file: Union[str, bytes, os.PathLike]) -> Dict:
    """Converts a yaml file into a dictionary"""
    with open(yaml_file, 'r') as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def read_config(filename: Union[str, bytes, os.PathLike]) -> Dict:
    """Reads the yaml configuration from file and returns it as a dictionary"""
    return yaml2dict(filename)


def generate_image(config_or_yaml: Dict or str, particle_data: dict = None, **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Generates a particle image based on a config (file). The generated image and the
    paticle data as a dictionary is returned.
    Particle positions are generated randomly or may be set via particle_data
    (dictionary with keys x, y, z, s).
    In the latter case, the density
    set in the config is ignored
    """
    if isinstance(config_or_yaml, dict):
        config = config_or_yaml
    else:
        config = yaml2dict(config_or_yaml)

    # read and process configuration:
    if config['square_image']:
        config['ny'] = config['nx']
    image_shape = (config['ny'], config['nx'])
    image_size = image_shape[0] * image_shape[1]

    sensitivity = config['sensitivity']
    qe = config['qe']
    dark_noise = config['dark_noise']
    baseline = config['noise_baseline']
    shot_noise_enabled = config['shot_noise']

    bit_depth = config['bit_depth']

    if particle_data is not None:
        xp = particle_data['x']
        yp = particle_data['y']
        zp = particle_data['z']
        psizes = particle_data['size']
        if any([_arg is not None for _arg in (xp, yp, zp, psizes)]):
            if not all([_arg is not None for _arg in (xp, yp, zp, psizes)]):
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
            if not isinstance(psizes, np.ndarray):
                if isinstance(psizes, (int, float)):
                    psizes = np.array([psizes])
                else:
                    psizes = np.asarray(psizes)
            # particle positions are set manually:
            for _arg in (xp, yp, zp, psizes):
                if _arg.ndim != 1:
                    raise ValueError(f'particle information must be 1D and not {_arg.ndim}D: {_arg}')
            n_particles = xp.size
    else:
        if config['particle_number'] < 1:
            raise ValueError('Argument "particle_number" invalid. Must be an integer greater than 0: '
                             f'{config["particle_number"]}')
        n_particles = int(config['particle_number'])
        assert n_particles > 0
        pmean = config['particle_size_mean']  # mean particle size
        pstd = config['particle_size_std']  # standard deviation of particle size
        pmin = pmean - 3 * pstd  # min particle size is 3*sigma below mean psize
        if pmin < PMIN_ALLOWED:
            warnings.warn(f'Particles smaller then {PMIN_ALLOWED} are set to {PMIN_ALLOWED}.')
            pmin = PMIN_ALLOWED
            if pmean <= pmin:
                raise ValueError(f'Mean particle size must be larger than smallest particle size!')
        pmax = pmean + 3 * pstd  # max particle size is 3*sigma above mean psize

    ppp = n_particles / image_size  # real ppp

    q = 2 ** bit_depth
    dz0 = config['laser_width']
    s = config['laser_shape_factor']

    # seed particles:
    _intensity = np.zeros(image_shape)
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
        # psizes = np.clip(np.random.normal(pmean, pstd, n_particles), pmin, pmax)
        # but rather redo the normal distbution for the outliers:
        psizes = np.random.normal(pmean, pstd, n_particles)
        iout = np.argwhere((psizes < pmin) | (psizes > pmax))
        for i in iout[:, 0]:
            dp = np.random.normal(pmean, pstd)
            while dp < pmin or dp > pmax:
                dp = np.random.normal(pmean, pstd)
            psizes[i] = dp

    # illuminate:
    if config['particle_size_illumination_dependency']:
        part_intensity = particle_intensity(zp, dz0, s, dp=psizes)
        part_intensity = part_intensity / (np.pi * max(psizes) ** 2 / 8)
    else:
        part_intensity = particle_intensity(zp, dz0, s)
    part_intensity = part_intensity * q * config['sensor_gain']
    ny, nx = image_shape
    # nsigma = 4
    for x, y, z, psize, pint in zip(xp, yp, zp, psizes, part_intensity):
        delta = int(10 * psize)
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
        _intensity[ymin:ymax, xmin:xmax] += np.exp(-8 * squared_dist / psize ** 2) * pint

    # add noise (pass the number of photons! That's why multiplication with sensitivity)
    if dark_noise == 0. and not shot_noise_enabled and baseline == 0.:
        pass  # add NO noise
    else:
        _intensity = add_camera_noise(_intensity / sensitivity, qe=qe, sensitivity=sensitivity,
                                      dark_noise=dark_noise, baseline=baseline, enable_shot_noise=shot_noise_enabled)

    max_adu = int(2 ** bit_depth - 1)
    _saturated_pixels = _intensity > max_adu
    n_saturated_pixels = np.sum(_saturated_pixels)

    _intensity[_saturated_pixels] = max_adu  # models pixel saturation

    attrs = {'bit_depth': bit_depth,
             'noise_baseline': baseline,
             'noise_darknoise': dark_noise,
             'noise_eq': qe,
             'noise_sensitivity': sensitivity,
             'n_saturated_pixels': n_saturated_pixels,
             # 'ps_mean': np.mean(psizes),
             # 'ps_std': np.std(psizes),
             'ppp': ppp,
             'n_particles': n_particles,
             'laser_width': dz0,
             'laser_shape_factor': s,
             'laser_max_intensity': q,
             'particle_size_mean': config['particle_size_mean'],
             'particle_size_std': config['particle_size_std'],
             'sensor_gain': config['sensor_gain'],
             'code_source': 'https://git.scc.kit.edu/da4323/piv-particle-density',
             'version': __version__}

    return _intensity.astype(int), attrs, {'x': xp, 'y': yp, 'z': zp, 'size': psizes, 'intensity': part_intensity}


# def combine_particle_image_data_arrays(datasets: List[xr.DataArray]) -> xr.DataArray:
#     """Combines the xr.DataArrays to a single DataArray"""
#     # _ = xr.DataArray(dims='image_index', data=np.arange(1, len(datasets) + 1))
#     ni = xr.DataArray(name='n_particles', dims='image_index', data=[d.n_particles for d in datasets])
#     ps_mean = xr.DataArray(name='ps_mean', dims='image_index', data=[d.ps_mean for d in datasets])
#     ps_std = xr.DataArray(name='ps_std', dims='image_index', data=[d.ps_std for d in datasets])
#     noise_baseline = xr.DataArray(name='noise_baseline', dims='image_index',
#                                   data=[d.noise_baseline for d in datasets])
#     noise_darknoise = xr.DataArray(name='dark_noise', dims='image_index',
#                                    data=[d.noise_darknoise for d in datasets])
#     n_saturated_pixels = xr.DataArray(name='n_saturated_pixels ', dims='image_index',
#                                       data=[d.n_saturated_pixels for d in datasets])
#
#     attrs = {k: v for k, v in datasets[0].attrs.items() if k not in ('n_particles', 'ps_mean',
#                                                                      'ps_std', 'noise_mean',
#                                                                      'noise_std', 'n_saturated_pixels',
#                                                                      'ppp')}
#
#     intensity = np.empty((len(datasets), *datasets[0].shape))
#     for i in range(len(datasets)):
#         intensity[i, ...] = datasets[i].values
#
#     return xr.DataArray(name='intensity', dims=('image_index', 'y', 'x'), data=intensity,
#                         coords={'ps_mean': ps_mean, 'ps_std': ps_std, 'n_particles': ni,
#                                 'noise_baseline': noise_baseline,
#                                 'noise_darknoise': noise_darknoise,
#                                 'n_saturated_pixels': n_saturated_pixels},
#                         attrs=attrs)


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
    """
    if s == 0:
        if dp is None:
            return np.ones_like(z)
        else:
            return np.ones_like(z) * dp ** 2 * np.pi / 8
    if dp is None:
        return np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s)
    else:
        return np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s) * dp ** 2 * np.pi / 8


def _generate_images_and_store_to_nc(cfg: Dict, n: int,
                                     data_directory: Union[str, bytes, os.PathLike]) -> None:
    _data_dir = Path(data_directory)
    if not _data_dir.is_dir():
        os.mkdir(_data_dir)

    for i in range(n):
        _intensity, _partpos = generate_image(cfg)
        save_dataset(_data_dir.joinpath(f'ds{cfg["fname"]}_{i:04d}.nc'), _intensity, _partpos)


def _generate(cfgs: List[Dict], nproc: int) -> Tuple[np.ndarray, List[Dict]]:
    """Generates the particle image(s) and returns those alongside with the particle
    information hidden in the image(s)

    Parameters
    ----------
    cfgs: List[Dict]
        List of configuration to be passed to `generate_image`
    nproc: int, default=CPU_COUNT
        Number of prcessors to be used to generate the data

    Returns
    -------
    intensities: np.ndarray
        PIV image of size (n, ny, nx) where n=number of images and image size (ny, nx)
    particle_information: List[Dict]
        List of dictionary with the information like particle position, size, ... for the
        generated images
    """
    if isinstance(cfgs, Dict):
        cfgs = [cfgs, ]

    if nproc > CPU_COUNT:
        warnings.warn('The number of processors you provided is larger than the '
                      f'maximum of your computer. Will continue with {CPU_COUNT} processors instead.')
        _nproc = CPU_COUNT
    else:
        _nproc = nproc
    if cfgs[0]['square_image']:
        cfgs[0]['ny'] = cfgs[0]['nx']
    intensities = np.empty(shape=(len(cfgs), cfgs[0]['ny'], cfgs[0]['nx']))
    # intensities = xr.DataArray(name='intensity', dims=('y', 'x'),
    #                            data=np.empty(shape=(len(cfgs), cfgs[0]['ny'], cfgs[0]['nx'])))
    particle_information = []
    attrs_ls = []
    if _nproc < 2:
        idx = 0
        for _cfg in tqdm(cfgs, total=len(cfgs), unit='cfg dict'):
            particle_data = process_config_for_particle_position(_cfg)

            _intensity, _attrs, _partpos = generate_image(_cfg, particle_data=particle_data)
            intensities[idx, ...] = _intensity
            attrs_ls.append(_attrs)
            particle_information.append(_partpos)
            idx += 1
        return intensities, attrs_ls, particle_information
    else:
        with mp.Pool(processes=_nproc) as pool:
            results = [pool.apply_async(generate_image, args=(_cfg, process_config_for_particle_position(_cfg))) for
                       _cfg in cfgs]
            for i, r in tqdm(enumerate(results), total=len(results)):
                intensity, _attrs, particle_meta = r.get()
                intensities[i, ...] = intensity
                attrs_ls.append(_attrs)
                particle_information.append(particle_meta)
            return intensities, attrs_ls, particle_information


@dataclass
class ConfigManager:
    """Configuration class which manages creation of images and labels from one or multiple configurations"""
    cfgs: Dict

    def __post_init__(self):
        if isinstance(self.cfgs, Dict):
            self.cfgs = (self.cfgs,)

    def __repr__(self):
        return f'Configurations with {len(self.cfgs)} configurations'

    def __len__(self):
        return len(self.cfgs)

    def generate(self, nproc: int = CPU_COUNT) -> Tuple[np.ndarray, np.ndarray]:
        """returns the generated data (intensities and particle information)
        This will not return all particle image information. Only number of particles!"""
        return _generate(self.cfgs, nproc)

    def to_hdf(self, data_directory: Union[str, bytes, os.PathLike],
               overwrite: bool = False, nproc: int = CPU_COUNT,
               compression: str = 'gzip', compression_opts: int = 5,
               n_split: int = 10000) -> List[Path]:
        """
        Generates the images and writes data in chunks to multiple files according to chunking.
        Besides the generated image, the following meta informations are also stored with the intention
        that they will be used as labels:
        - number of particles
        - particle size mean
        - particle size std
        - intensity mean
        - intensity std

        Files are stored in data_directory and are named ds_XXXXXX.hdf where XXXXXX is the index of
        the file.

        Parameters
        ----------
        data_directory: str, bytes, os.PathLike
            Path to directory where HDF5 files should be stored.
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
        _dir = Path(data_directory)
        _ds_filenames = list(_dir.glob('ds*.hdf'))
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
        for ichunk, cfg_chunk in enumerate(chunked_cfgs):
            images, attrs, particle_information = _generate(cfg_chunk, nproc)
            new_name = f'ds_{ichunk:06d}.hdf'
            new_filename = _dir.joinpath(new_name)
            filenames.append(new_filename)
            n_ds, ny, nx = images.shape
            with h5py.File(new_filename, 'w') as h5:
                ds_imageindex = h5.create_dataset('image_index', data=np.arange(0, n_ds, 1), dtype=int)
                ds_imageindex.attrs['long_name'] = 'image index'
                ds_imageindex.attrs['units'] = ''
                ds_imageindex.make_scale()

                ds_x_pixel_coord = h5.create_dataset('ix', data=np.arange(0, nx, 1), dtype=int)
                ds_x_pixel_coord.attrs['standard_name'] = 'x_pixel_coordinate'
                ds_x_pixel_coord.attrs['units'] = 'px'
                ds_x_pixel_coord.make_scale()

                ds_y_pixel_coord = h5.create_dataset('iy', data=np.arange(0, ny, 1), dtype=int)
                ds_y_pixel_coord.attrs['standard_name'] = 'y_pixel_coordinate'
                ds_y_pixel_coord.attrs['units'] = 'px'
                ds_y_pixel_coord.make_scale()

                ds_images = h5.create_dataset('images', shape=images.shape, compression=compression,
                                              compression_opts=compression_opts,
                                              chunks=(1, *images.shape[1:]))
                ds_images.attrs['long_name'] = 'image intensity'
                ds_images.attrs['units'] = 'counts'

                ds_nparticles = h5.create_dataset('nparticles', shape=n_ds,
                                                  compression=compression,
                                                  compression_opts=compression_opts, dtype=int)
                ds_nparticles.attrs['long_name'] = 'number of particles'
                ds_nparticles.attrs['units'] = ''
                ds_nparticles.make_scale()

                ds_particledens = h5.create_dataset('particle_density', shape=n_ds,
                                                    compression=compression,
                                                    compression_opts=compression_opts, dtype=float)
                ds_particledens.attrs['long_name'] = 'particle density'
                ds_particledens.attrs['units'] = '1/px'
                ds_particledens.make_scale()

                ds_mean_size = h5.create_dataset('particle_size_mean', shape=n_ds, compression=compression,
                                                 compression_opts=compression_opts)
                ds_mean_size.attrs['units'] = 'px'
                ds_mean_size.make_scale()

                ds_configured_mean_size = h5.create_dataset('configured_particle_size_mean', shape=n_ds,
                                                            compression=compression,
                                                            compression_opts=compression_opts)
                ds_configured_mean_size.attrs['units'] = 'px'
                ds_configured_mean_size.make_scale()

                ds_std_size = h5.create_dataset('particle_size_std', shape=n_ds, compression=compression,
                                                compression_opts=compression_opts)
                ds_std_size.attrs['units'] = 'px'
                ds_std_size.make_scale()

                ds_configured_std_size = h5.create_dataset('configured_particle_size_std', shape=n_ds,
                                                           compression=compression,
                                                           compression_opts=compression_opts)
                ds_configured_std_size.attrs['units'] = 'px'
                ds_configured_std_size.make_scale()

                ds_intensity_mean = h5.create_dataset('particle_intensity_mean', shape=n_ds, compression=compression,
                                                      compression_opts=compression_opts)
                ds_intensity_mean.attrs['units'] = 'counts'
                ds_intensity_mean.make_scale()

                ds_intensity_std = h5.create_dataset('particle_intensity_std', shape=n_ds, compression=compression,
                                                     compression_opts=compression_opts)
                ds_intensity_std.attrs['units'] = 'counts'
                ds_intensity_std.make_scale()

                ds_n_satpx = h5.create_dataset('number_of_saturated_pixels', shape=n_ds, compression=compression,
                                               compression_opts=compression_opts)
                ds_n_satpx.attrs['units'] = 'counts'
                ds_n_satpx.make_scale()

                ds_laser_width = h5.create_dataset('laser_width', shape=n_ds, compression=compression,
                                                   compression_opts=compression_opts)
                ds_laser_width.attrs['units'] = 'm'
                ds_laser_width.make_scale()

                ds_bitdepth = h5.create_dataset('bit_depth', shape=n_ds, compression=compression,
                                                compression_opts=compression_opts, dtype=int)
                ds_bitdepth.attrs['units'] = ''
                ds_bitdepth.make_scale()

                ds_laser_shape_factor = h5.create_dataset('laser_shape_factor', shape=n_ds, compression=compression,
                                                          compression_opts=compression_opts)
                ds_laser_shape_factor.attrs['units'] = ''
                ds_laser_shape_factor.make_scale()

                ds_n_satpx[:] = [a['n_saturated_pixels'] for a in attrs]
                ds_laser_shape_factor[:] = [a['laser_shape_factor'] for a in attrs]
                ds_laser_width[:] = [a['laser_width'] for a in attrs]

                ds_images[:] = images
                npart = np.asarray([len(p['x']) for p in particle_information])
                ds_nparticles[:] = npart
                ds_particledens[:] = npart / (nx * ny)
                ds_mean_size[:] = [np.mean(p['size']) for p in particle_information]
                ds_configured_mean_size[:] = [a['particle_size_mean'] for a in attrs]
                ds_configured_std_size[:] = [a['particle_size_std'] for a in attrs]
                ds_std_size[:] = [np.std(p['size']) for p in particle_information]
                ds_intensity_mean[:] = [np.mean(p['intensity']) for p in particle_information]
                ds_intensity_std[:] = [np.std(p['intensity']) for p in particle_information]
                ds_bitdepth[:] = [a['bit_depth'] for a in attrs]

                for ds in (ds_imageindex, ds_nparticles, ds_mean_size, ds_std_size,
                           ds_intensity_mean, ds_intensity_std,
                           ds_laser_width, ds_laser_shape_factor, ds_n_satpx,
                           ds_particledens, ds_bitdepth,
                           ds_configured_mean_size,
                           ds_configured_std_size):
                    ds_images.dims[0].attach_scale(ds)
                ds_images.dims[1].attach_scale(ds_y_pixel_coord)
                ds_images.dims[2].attach_scale(ds_x_pixel_coord)

                sntt = StandardNameTableTranslation.from_yaml(
                    pathlib.Path(__file__).parent / 'synpivimage-to-synpiv-v1.yml')
                sntt.translate_group(h5)
                print('... done.')
        return filenames


def build_ConfigManager(initial_cfg: Dict,
                        variations: List[Tuple[str, Union[float, np.ndarray, Tuple]]],
                        per_combination: int = 1, shuffle: bool = True) -> ConfigManager:
    """Generates a list of configuration dictionaries.
    Request an initial configuration and a tuple of variable length containing
    the name of a dictionary key of the configuraiton and the values to be chosen.
    A list containing configuration dictionaries of all combinations is
    returned. Moreover, a filename is generated and added to the dictionary.

    Parameters
    ----------
    initial_cfg : Dict
        Initial configuration to take and replace parameters to vary in
    variations: List
        List of Tuples describing what to vary. An entry in the list
        must look like ('parameter_name', [1, 2, 3])
    per_combination: int=1
        Number of configurations per parameter set (It may be useful to repeat
        the generation of a specific parameter set because particles are randomly
        generated. Default is 1.
    shuffle: bool=True
        Shuffle the config files. Default is True.

    """

    variation_dict = {n: v for n, v in variations}

    # if variation has a float entry, make it a list:
    for k, v in variation_dict.items():
        if isinstance(v, (int, float)):
            variation_dict[k] = [v, ]

    keys, values = zip(*variation_dict.items())
    _dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    cfgs = []
    count = 0
    for _, param_dict in enumerate(_dicts):
        for icomb in range(per_combination):
            cfgs.append(initial_cfg.copy())
            for k, v in param_dict.items():
                cfgs[-1][k] = v
                # a a raw filename without extension to the dictionary:
            cfgs[-1]['fname'] = f'ds{count:06d}'
            count += 1
    if shuffle:
        random.shuffle(cfgs)
    return ConfigManager(cfgs)


if __name__ == '__main__':
    pass
