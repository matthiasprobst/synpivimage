import logging
import pathlib

from ._version import __version__
from .camera import Camera
from .core import take_image
from .io import Imwriter, HDF5Writer
from .laser import Laser
from .particles import Particles

__this_dir__ = pathlib.Path(__file__).parent

logging.basicConfig()
logger = logging.getLogger(__package__)

_formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S'
)

_sh = logging.StreamHandler()
_sh.setFormatter(_formatter)
logger.addHandler(_sh)


def set_loglevel(level):
    """Set the log level"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


set_loglevel(logging.INFO)

__all__ = ['__version__', 'Camera', 'take_image', 'Laser', 'Particles', 'set_loglevel']

__package_dir__ = pathlib.Path(__file__).parent
