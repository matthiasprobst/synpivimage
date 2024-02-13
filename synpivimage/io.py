import cv2
import h5py
import json
import numpy as np
import pathlib
from typing import Dict, Union, List

from . import get_software_source_code_meta
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
    meta['software_info'] = get_software_source_code_meta()
    return meta


def hdfwrite(group: h5py.Group,
             img: Union[np.ndarray, List[np.ndarray]],
             camera: Camera,
             laser: Laser,
             particles: Union[Particles, List[Particles]],
             img_dataset_name='img'):
    """Write to open HDF5 file"""
    group.attrs['software_info'] = json.dumps(get_software_source_code_meta())
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
    imgds = group.create_dataset(img_dataset_name, shape=(n_imgs, *img_shape))
    for i, im in enumerate(imgs):
        imgds[i, ...] = im
    metadata = group.create_group('virtual_piv_setup')

    def _write_component(_component, _group):
        for k, v in _component.model_dump().items():
            _group.create_dataset(k, data=v)

    for component in (camera, laser, particles):
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
