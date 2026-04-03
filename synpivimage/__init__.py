import logging
import pathlib

from ._version import __version__
from .camera import Camera
from .core import take_image
from .io import Imwriter, HDF5Writer
from .laser import Laser
from .particles import Particles

__this_dir__ = pathlib.Path(__file__).parent

logger = logging.getLogger("synpivimage")

_formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S'
)

def _configure_default_logging() -> None:
    """Configure package logger once without mutating root logging."""
    if logger.handlers:
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_formatter)
    logger.addHandler(stream_handler)


def set_loglevel(level: int) -> None:
    """Set the log level"""
    _configure_default_logging()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


_configure_default_logging()
set_loglevel(logging.INFO)

__all__ = ['__version__', 'Camera', 'take_image', 'Laser', 'Particles', 'set_loglevel']

__package_dir__ = pathlib.Path(__file__).parent
