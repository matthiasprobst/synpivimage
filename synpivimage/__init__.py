import pathlib

from ._version import __version__
from .core import (generate_default_yaml_file,
                   generate_image,
                   DEFAULT_CFG,
                   build_ConfigManager,
                   ConfigManager)

__package_dir__ = pathlib.Path(__file__).parent
generate_default_yaml_file()

__all__ = ['__version__', 'build_ConfigManager', 'ConfigManager',
           'DEFAULT_CFG', 'generate_image', 'generate_default_yaml_file',
           '__package_dir__']
