import cv2
import json
import numpy as np
import pathlib
from typing import Dict, Union


def imwrite(filename: Union[str, pathlib.Path],
            img: np.ndarray,
            overwrite: bool = False,
            meta: Dict = None,
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
    plib_filename = pathlib.Path(filename)
    if plib_filename.exists():
        if not overwrite:
            raise FileExistsError(f"File {filename} exists and overwrite is False")
        else:
            plib_filename.unlink()
    cv2.imwrite(str(filename), img, **kwargs)
    if meta is not None:
        meta_filename = plib_filename.with_suffix('.json')
        metawrite(meta_filename, meta)
    return plib_filename


def imwrite16(filename,
              img,
              overwrite: bool = False,
              meta: Dict = None, **kwargs):
    """Write an image to a 16 bit file"""
    return imwrite(filename, img.astype(np.uint16), overwrite, meta=meta, **kwargs)


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
