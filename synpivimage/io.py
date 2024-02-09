import cv2
import h5py
import json
import numpy as np
import pathlib
from typing import Dict, Union

from .camera import Camera
from .component import Component
from .laser import Laser


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
        if isinstance(v, Component):
            meta[k] = v.model_dump()
        else:
            meta[k] = v
    return meta


def hdfwrite(dataset: h5py.Dataset,
             index: int,
             img: np.ndarray,
             camera: Camera,
             laser: Laser,
             meta_group: Union[h5py.Group, str] = 'meta') -> None:
    """Write to open HDF5 file. The target dataset must have been
    prepared correctly!"""
    dataset[index, ...] = img[:]
    if isinstance(meta_group, str):
        meta_group = dataset.parent[meta_group]
    for component in (camera, laser):
        if component:
            comp_grp = meta_group[component.__class__.__name__]
            for k, v in component.model_dump().items():
                comp_grp[k][index] = v


def imwrite(filename: Union[str, pathlib.Path],
            img: np.ndarray,
            overwrite: bool = False,
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
    if len(kwargs) > 0:
        meta = build_meta_dict(**kwargs)

        with open(plib_filename.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=4)

    return plib_filename


def imwrite16(filename,
              img,
              overwrite: bool = False,
              meta: Dict = None, **kwargs):
    """Write an image to a 16 bit file"""
    return imwrite(filename,
                   img.astype(np.uint16),
                   overwrite,
                   meta=meta,
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


def metawrite(filename, meta: Dict):
    """Write metadata to a json file"""
    filename = pathlib.Path(filename)
    meta_filename = filename.with_suffix('.json')
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
