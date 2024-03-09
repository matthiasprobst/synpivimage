import logging
import pathlib

from ._version import __version__
from .camera import Camera
from .core import take_image
from .laser import Laser
from .particles import Particles

__this_dir__ = pathlib.Path(__file__).parent

logging.basicConfig()
logger = logging.getLogger(__package__)
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
logger.addHandler(_sh)

__all__ = ['__version__', 'Camera', 'take_image', 'Laser', 'Particles']

__package_dir__ = pathlib.Path(__file__).parent
