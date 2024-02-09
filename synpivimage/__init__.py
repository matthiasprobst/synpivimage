import pathlib

from ._version import __version__
from .core import take_image
from .plotting import imshow

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
