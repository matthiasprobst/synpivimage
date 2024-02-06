import cv2
import numpy as np
import pathlib
from typing import Union


def imwrite(filename: Union[str, pathlib.Path],
            img: np.ndarray,
            overwrite: bool = False, **kwargs) -> pathlib.Path:
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
    return plib_filename


def imwrite16(filename, img, overwrite: bool = False, **kwargs):
    """Write an image to a 16 bit file"""
    return imwrite(filename, img.astype(np.uint16), overwrite, **kwargs)


def imwrite8(filename, img, overwrite: bool = False, **kwargs):
    """Write an image to a 8 bit file"""
    return imwrite(filename, img.astype(np.uint8), overwrite, **kwargs)


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
