import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Dict, List, Union, Tuple

import h5py
import numpy as np
import xarray as xr
import yaml
from tqdm.notebook import tqdm

from ._version import __version__
from .noise import add_camera_noise

np.random.seed()
CPU_COUNT = mp.cpu_count()

default_yaml_file = 'default.yaml'

PMIN_ALLOWED = 0.1

DEFAULT_CFG = {'ny': 128, 'nx': 128, 'bit_depth': 16,
                  'noise_baseline': 20, 'dark_noise': 2.29,
                  'sensitivity': 1, 'qe': 1, 'shot_noise': True,
                  'particle_number': 0.1, 'particle_size_mean': 2.5,
                  'particle_size_std': 0.25,
                  'laser_width': 3, 'laser_shape_factor': 4
                  }


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
        n_particles = int(config['particle_number'])
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

    q = config['laser_max_intensity']
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
        psizes = np.clip(np.random.normal(pmean, pstd, n_particles), pmin, pmax)

    # illuminate:
    part_intensity = particle_intensity(zp, q, dz0, s)
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
    _intensity = add_camera_noise(_intensity / sensitivity, qe=qe, sensitivity=sensitivity,
                                  dark_noise=dark_noise, baseline=baseline, enable_shot_noise=shot_noise_enabled)

    max_adu = np.int(2 ** bit_depth - 1)
    _saturated_pixels = _intensity > max_adu
    n_saturated_pixels = np.sum(_saturated_pixels)

    _intensity[_saturated_pixels] = max_adu  # models pixel saturation

    attrs = {'bit_depth': bit_depth,
             'noise_baseline': baseline,
             'noise_darknoise': dark_noise,
             'noise_eq': qe,
             'noise_sensitivity': sensitivity,
             'n_saturated_pixels': n_saturated_pixels,
             'ps_mean': np.mean(psizes),
             'ps_std': np.std(psizes),
             'ppp': ppp,
             'n_particles': n_particles,
             'laser_width': dz0,
             'laser_shape_factor': s,
             'laser_max_intensity': q,
             'code_source': 'https://git.scc.kit.edu/da4323/piv-particle-density',
             'version': __version__}

    image_data_array = xr.DataArray(name='intensity', dims=('y', 'x'), data=_intensity.astype(int),
                                    attrs=attrs)
    return image_data_array, {'x': xp, 'y': yp, 'z': zp, 'size': psizes, 'intensity': part_intensity}


def combine_particle_image_data_arrays(datasets: List[xr.DataArray]) -> xr.DataArray:
    """Combines the xr.DataArrays to a single DataArray"""
    # _ = xr.DataArray(dims='image_index', data=np.arange(1, len(datasets) + 1))
    ni = xr.DataArray(name='n_particles', dims='image_index', data=[d.n_particles for d in datasets])
    ps_mean = xr.DataArray(name='ps_mean', dims='image_index', data=[d.ps_mean for d in datasets])
    ps_std = xr.DataArray(name='ps_std', dims='image_index', data=[d.ps_std for d in datasets])
    noise_baseline = xr.DataArray(name='noise_baseline', dims='image_index',
                                  data=[d.noise_baseline for d in datasets])
    noise_darknoise = xr.DataArray(name='dark_noise', dims='image_index',
                                   data=[d.noise_darknoise for d in datasets])
    n_saturated_pixels = xr.DataArray(name='n_saturated_pixels ', dims='image_index',
                                      data=[d.n_saturated_pixels for d in datasets])

    attrs = {k: v for k, v in datasets[0].attrs.items() if k not in ('n_particles', 'ps_mean',
                                                                     'ps_std', 'noise_mean',
                                                                     'noise_std', 'n_saturated_pixels',
                                                                     'ppp')}

    intensity = np.empty((len(datasets), *datasets[0].shape))
    for i in range(len(datasets)):
        intensity[i, ...] = datasets[i].values

    return xr.DataArray(name='intensity', dims=('image_index', 'y', 'x'), data=intensity,
                        coords={'ps_mean': ps_mean, 'ps_std': ps_std, 'n_particles': ni,
                                'noise_baseline': noise_baseline,
                                'noise_darknoise': noise_darknoise,
                                'n_saturated_pixels': n_saturated_pixels},
                        attrs=attrs)


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


def particle_intensity(z: np.ndarray, q: float, dz0: float, s: int):
    """Intensity of a particle in the laser beam.

    Parameters
    ----------
    z : array-like
        particle z-position
    q : float
        laser max_intensity
    dz0 : float
        laser beam width. for 2 profile is gaussian
    s : int
        shape factor
    """
    # Note: 0.5*dz0 because width is given not "distance to center"!
    if s == 0:
        return q * np.ones_like(z)

    return q * np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s)


# def imwrite(fpath, arr, bit_depth):
#     if isinstance(arr, xr.DataArray):
#         _arr = arr.values
#     else:
#         _arr = arr
#     tifffile.imwrite(fpath, _arr.astype('uint16'))


def save_dataset(path, intensity, particle_data):
    """writes a xarray Dataset to file (nc)"""
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d_%H:%M:%S')
    dset = xr.Dataset({'intensity': intensity,
                       'particle_xpos': xr.DataArray(dims='particle_index', data=particle_data['x']),
                       'particle_ypos': xr.DataArray(dims='particle_index', data=particle_data['y']),
                       'particle_zpos': xr.DataArray(dims='particle_index', data=particle_data['z']),
                       'particle_size': xr.DataArray(dims='particle_index', data=particle_data['size']),
                       'particle_intensity': xr.DataArray(dims='particle_index', data=particle_data['intensity'])})
    dset.attrs = {'code_source': 'https://git.scc.kit.edu/da4323/piv-particle-density',
                  'code_version': __version__, 'creation_time': dt_string}
    dset.to_netcdf(path)


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

    intensities = np.empty(shape=(len(cfgs), cfgs[0]['ny'], cfgs[0]['nx']))
    particle_information = []
    if _nproc < 2:
        idx = 0
        for _cfg in tqdm(cfgs, total=len(cfgs), unit='cfg dict'):
            _intensity, _partpos = generate_image(_cfg)
            intensities[idx, ...] = _intensity
            particle_information.append(len(_partpos['x']))
            idx += 1
        return intensities, particle_information
    else:
        with mp.Pool(processes=_nproc) as pool:
            results = [pool.apply_async(generate_image, args=(_cfg,)) for _cfg in cfgs]
            for i, r in tqdm(enumerate(results), total=len(results)):
                intensity, particle_meta = r.get()
                intensities[i, ...] = intensity
                particle_information.append(particle_meta)
            return intensities, particle_information


def generate_and_write_images(cfg: Union[List[Dict], Dict], n: int,
                              data_directory: Union[str, bytes, os.PathLike],
                              nproc: int = CPU_COUNT) -> None:
    """Generates the image(s) according to the configuration(s) and returns raw data
    or writes data directly to a netCDF4 file

    Parameters
    ----------
    cfg: List[Dict] or Dict
        List of configuration dictionaries or a single config dictionary.
    n: int
        Number of re-runs of the provided configuration.
    data_directory: Union[str, bytes, os.PathLike]
        Directory path where to write data to.
    nproc: int, default=CPU_COUNT
        Number of processors to take for multiprocessing.

    """
    if nproc > CPU_COUNT:
        warnings.warn('The number of processors you provided is larger than the '
                      f'maximum of your computer. Will continue with {CPU_COUNT} processors instead.')
        _nproc = CPU_COUNT
    else:
        _nproc = nproc

    if isinstance(cfg, dict):
        _generate_images_and_store_to_nc(cfg, data_directory)
        return

    if _nproc < 2:
        for _cfg in tqdm(cfg):
            _generate_images_and_store_to_nc(_cfg, data_directory)
    else:
        with mp.Pool(processes=_nproc) as pool:
            results = [pool.apply_async(_generate_images_and_store_to_nc, args=(_cfg, data_directory)) for _cfg in
                       cfg]

            for r in tqdm(results):
                _ = r.get()


@dataclass
class Configurations:
    """Configuration class which manages creation of images and labels from one or multiple configurations"""
    cfgs: Dict

    def __post_init__(self):
        if isinstance(self.cfgs, Dict):
            self.cfgs = (self.cfgs,)

    def __repr__(self):
        return f'Configurations with {len(self.cfgs)} configurations'

    def __len__(self):
        return len(self.cfgs)

    def generate(self, n: int, nproc: int = CPU_COUNT) -> Tuple[np.ndarray, np.ndarray]:
        """returns the generated data (intensities and particle information)
        This will not return all particle image information. Only number of particles!"""
        return _generate(self.cfgs, nproc)

    def to_hdf(self, n: int, data_directory: Union[str, bytes, os.PathLike],
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
        n: int
            Number of datasets per parameter
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
            chunked_cfgs = [self.cfgs * n, ]
        else:
            chunked_cfgs = _chunk(self.cfgs * n, n_split)
            _nfiles = ceil(len(self.cfgs) / n_split * n)
        print('Writing HDF5 file(s). This may take a while... '
              f'(files to write: {_nfiles}. datasets to write in total: {len(self.cfgs * n)})')

        filenames = []
        for ichunk, cfg_chunk in enumerate(chunked_cfgs):
            images, particle_information = _generate(cfg_chunk, nproc)
            new_name = f'ds_{ichunk:06d}.hdf'
            new_filename = _dir.joinpath(new_name)
            filenames.append(new_filename)
            n_ds = images.shape[0]
            with h5py.File(new_filename, 'w') as h5:
                ds_images = h5.create_dataset('image', shape=images.shape, compression=compression,
                                              compression_opts=compression_opts,
                                              chunks=(1, *images.shape[1:]))
                ds_nparticles = h5.create_dataset('nparticles', shape=n_ds,
                                                  compression=compression,
                                                  compression_opts=compression_opts)
                ds_mean_size = h5.create_dataset('particle_size_mean', shape=n_ds, compression=compression,
                                                 compression_opts=compression_opts)
                ds_std_size = h5.create_dataset('particle_size_std', shape=n_ds, compression=compression,
                                                compression_opts=compression_opts)
                ds_intensity_mean = h5.create_dataset('particle_intensity_mean', shape=n_ds, compression=compression,
                                                      compression_opts=compression_opts)
                ds_intensity_std = h5.create_dataset('particle_intensity_std', shape=n_ds, compression=compression,
                                                     compression_opts=compression_opts)
                ds_nparticles.attrs['long_name'] = 'number of particles'
                ds_images.attrs['long_name'] = 'image intensities'
                ds_images[:] = images
                ds_nparticles[:] = [len(p['x']) for p in particle_information]
                ds_mean_size[:] = [np.mean(p['size']) for p in particle_information]
                ds_std_size[:] = [np.std(p['size']) for p in particle_information]
                ds_intensity_mean[:] = [np.mean(p['intensity']) for p in particle_information]
                ds_intensity_std[:] = [np.std(p['intensity']) for p in particle_information]
                print('... done.')
        return filenames

    def to_nc(self, n: int, data_directory: Union[str, bytes, os.PathLike], nproc=CPU_COUNT) -> None:
        """Generates the image(s) according to the configuration(s) and returns raw data
        or writes data directly to a netCDF4 file

        Parameters
        ----------
        n: int
            Number of re-runs of the provided configuration.
        data_directory: Union[str, bytes, os.PathLike]
            Directory path where to write data to.
        nproc: int, default=CPU_COUNT
            Number of processors to take for multiprocessing.

        """
        if nproc > CPU_COUNT:
            warnings.warn('The number of processors you provided is larger than the '
                          f'maximum of your computer. Will continue with {CPU_COUNT} processors instead.')
            _nproc = CPU_COUNT
        else:
            _nproc = nproc

        if _nproc < 2:
            for _cfg in tqdm(self.cfgs):
                _generate_images_and_store_to_nc(_cfg, data_directory)
        else:
            with mp.Pool(processes=_nproc) as pool:
                results = [pool.apply_async(_generate_images_and_store_to_nc, args=(_cfg, data_directory)) for _cfg
                           in self.cfgs]

                for r in tqdm(results):
                    _ = r.get()


if __name__ == '__main__':
    pass
