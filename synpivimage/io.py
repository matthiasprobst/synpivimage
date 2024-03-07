import json
import pathlib
import warnings
from typing import Dict, Union, List

import cv2
import h5py
import numpy as np

from . import get_package_meta
from .camera import Camera
from .laser import Laser
from .particles import Particles


def parse_filename(filename: Union[str, pathlib.Path],
                   overwrite: bool) -> pathlib.Path:
    filename = pathlib.Path(filename)

    if filename.exists():
        if not overwrite:
            raise FileExistsError(f"File {filename} exists and overwrite is False")
        else:
            filename.unlink()
    return filename


def build_meta_dict(**kwargs) -> dict:
    """generates a dictionary from the keyword arguments"""
    meta = {}

    for k, v in kwargs.items():
        if v is not None:
            if hasattr(v, 'model_dump'):
                meta[k] = v.model_dump()
            else:
                meta[k] = v
    meta['software_info'] = get_package_meta()
    return meta


def hdfwrite(group: h5py.Group,
             img: Union[np.ndarray, List[np.ndarray]],
             camera: Camera,
             laser: Laser,
             particles: Union[Particles, List[Particles]],
             img_dataset_name='img',
             bit_depth=16):
    warnings.warn("hdfwrite is deprecated, use hdfwrite16 or hdfwrite8 instead", DeprecationWarning)
    return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=bit_depth)


def hdfwrite16(group: h5py.Group,
               img: Union[np.ndarray, List[np.ndarray]],
               camera: Camera,
               laser: Laser,
               particles: Union[Particles, List[Particles]],
               img_dataset_name='img'):
    return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=16)


def hdfwrite8(group: h5py.Group,
              img: Union[np.ndarray, List[np.ndarray]],
              camera: Camera,
              laser: Laser,
              particles: Union[Particles, List[Particles]],
              img_dataset_name='img'):
    return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=8)


def _hdfwrite(group: h5py.Group,
              img: Union[np.ndarray, List[np.ndarray]],
              camera: Camera,
              laser: Laser,
              particles: Union[Particles, List[Particles]],
              img_dataset_name,
              bit_depth):
    """Write to open HDF5 file"""
    if bit_depth == 8:
        _dtype = 'uint8'
    elif bit_depth == 16:
        _dtype = 'uint16'
    else:
        raise ValueError("bit_depth must be 8 or 16")
    group.attrs['software_info'] = json.dumps(get_package_meta())
    if isinstance(img, (list, tuple)):
        assert isinstance(particles, (list, tuple)), "particles must be a list or tuple if img is a list or tuple"
        assert len(img) == len(particles), "img and particles must have the same length"
        n_imgs = len(img)
        img_shape = img[0].shape
        imgs = [np.asarray(i) for i in img]
    else:
        n_imgs = 1
        img_shape = img.shape
        imgs = [np.asarray(img)]
        particles = [particles, ]
    imgds = group.create_dataset(img_dataset_name, shape=(n_imgs, *img_shape), dtype=_dtype)
    for i, im in enumerate(imgs):
        imgds[i, ...] = im
    metadata = group.create_group('virtual_piv_setup')

    def _write_component(_component, _group):
        for k, v in _component.model_dump().items():
            _group.create_dataset(k, data=v)

    ref_len = particles[0].x.size
    equal_length_particles = all(ref_len == p.x.size for p in particles)
    if equal_length_particles:
        laser_grp = metadata.create_group(particles[0].__class__.__name__)

        for ic, (k, v) in enumerate(particles[0].model_dump().items()):
            ds = laser_grp.create_dataset(k, shape=(n_imgs, ref_len))
            ds[0, :] = v

        for i, p in enumerate(particles):
            for ic, (k, v) in enumerate(p.model_dump().items()):
                laser_grp[k][i, :] = v

        iterator = (camera, laser,)
    else:
        iterator = (camera, laser, particles[0])
    for component in iterator:
        if component:
            if isinstance(component, (list, tuple)):
                for ic, c in enumerate(component):
                    comp_grp = metadata.create_group(c.__class__.__name__ + f'_{ic}')
                    _write_component(c, comp_grp)
            else:
                comp_grp = metadata.create_group(component.__class__.__name__)
                _write_component(component, comp_grp)


def imwrite(filename: Union[str, pathlib.Path],
            img: np.ndarray,
            overwrite: bool = False,
            camera: Camera = None,
            laser: Laser = None,
            particles: Particles = None,
            **kwargs) -> pathlib.Path:
    """Write an image to a file. Calls cv2.imwrite!

    Parameters
    ----------
    filename : str or pathlib.Path
        Name of the file.
    img : ndarray
        Image to write to the file.
    overwrite : bool
        If True, overwrite the file if it exists.
        Default is False.
    meta: bool=None
        If given writes json file with metadata
    kwargs : dict
        Additional parameters to pass to the writer.

    Returns
    -------
    filename: pathlib.Path
        Path to the file written.
    """
    plib_filename = parse_filename(filename, overwrite)
    cv2.imwrite(str(filename), img)

    metadata = {'camera': camera, 'laser': laser, 'particles': particles}
    metadata.update(kwargs)
    metawrite(plib_filename.with_suffix('.json'), metadata)

    return plib_filename


def metawrite(filename: Union[str, pathlib.Path],
              metadata: Dict) -> None:
    filename = pathlib.Path(filename)
    meta = build_meta_dict(**metadata)
    with open(filename.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=4)


def imwrite16(filename,
              img,
              overwrite: bool = False,
              camera: Camera = None,
              laser: Laser = None,
              particles: Particles = None,
              **kwargs):
    """Write an image to a 16 bit file"""
    return imwrite(filename,
                   img.astype(np.uint16),
                   overwrite,
                   camera=camera,
                   laser=laser,
                   particles=particles,
                   **kwargs)


def imwrite8(filename,
             img,
             overwrite: bool = False,
             meta: Dict = None,
             **kwargs):
    """Write an image to a 8 bit file"""
    return imwrite(filename,
                   img.astype(np.uint8),
                   overwrite,
                   meta=meta,
                   **kwargs)


def imread(filename, flags=cv2.IMREAD_GRAYSCALE):
    """Read an image from a file. Calls cv2.imread!

    Parameters
    ----------
    filename : str
        Name of the file.
    flags : int
        Flags for reading the image.

    Returns
    -------
    ndarray
        Image read from the file.
    """
    if not pathlib.Path(filename).exists():
        raise FileNotFoundError(f"File {filename} not found")
    return cv2.imread(str(filename), flags)


def imread16(filename):
    """Read an image from a 16 bit file"""
    return imread(filename, -1).astype(np.uint16)


def imread8(filename, flags=cv2.IMREAD_GRAYSCALE):
    """Read an image from a 8 bit file"""
    return imread(filename, flags).astype(np.uint8)


def metaread(filename) -> Dict:
    """Read metadata from a image or json file. if image file is given, it looks for a .json file with the same name"""
    filename = pathlib.Path(filename)
    if filename.suffix == '.json':
        meta_filename = filename
    else:
        meta_filename = pathlib.Path(filename).with_suffix('.json')
    if not meta_filename.exists():
        raise FileNotFoundError(f"File {meta_filename} not found")
    with open(meta_filename, 'r') as f:
        return json.load(f)

# def metawrite(filename: Union[str, pathlib.Path], metadata: Dict):
#     """Write metadata to a json file"""
#     filename = pathlib.Path(filename)
#     meta_filename = filename.with_suffix('.json')
#     with open(meta_filename, 'w') as f:
#         json.dump(metadata, f, indent=4)
