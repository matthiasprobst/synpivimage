import json
import logging
import pathlib

from ._version import __version__
from .camera import Camera
from .core import take_image
from .laser import Laser
from .particles import Particles
from .plotting import imshow

__this_dir__ = pathlib.Path(__file__).parent

logging.basicConfig()
logger = logging.getLogger(__package__)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)


def get_software_source_code_meta():
    """Reads codemeta.json and returns it as dict"""
    with open(__this_dir__ / '../codemeta.json', 'r') as f:
        codemeta = json.loads(f.read())
    return codemeta


__all__ = ['__version__', 'Camera', 'take_image', 'Laser', 'imshow', 'Particles']

__package_dir__ = pathlib.Path(__file__).parent

# generate_default_yaml_file()


# def displace(particle_infos: Union[List[ParticleInfo], ParticleInfo], dx=None, dy=None, dz=None):
#     """displaces one or multiple particles"""
#     if not isinstance(particle_infos, list):
#         return particle_infos.displace(dx, dy, dz)
#     for p in particle_infos:
#         p.displace(dx, dy, dz)
#
#
# def displace_from_hdf(hdf_filename, dx=None, dy=None, dz=None):
#     """displaces one or multiple particles"""
#     return displace(ParticleInfo.from_hdf(hdf_filename), dx, dy, dz)
#
#
# __all__ = ['__version__', 'build_ConfigManager', 'ConfigManager',
#            'get_default', 'generate_image', 'generate_default_yaml_file',
#            '__package_dir__']
