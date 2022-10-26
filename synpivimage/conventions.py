"""Layout created with h5RDMtoolbox"""

import pathlib

from h5rdmtoolbox.conventions import layout
from h5rdmtoolbox.conventions.cflike.standard_name import StandardNameTableTranslation
from h5rdmtoolbox.conventions.layout import H5Layout

__this_dir__ = pathlib.Path(__file__).parent


class StandardNameTranslation:
    """Interface class for SNT"""

    def __new__(cls, *args, **kwargs) -> StandardNameTableTranslation:
        return StandardNameTableTranslation.from_yaml(
            __this_dir__ / 'synpivimage-to-particle_image_velocimetry-v1.yml'
        )


class Layout:
    """Layout class"""

    def __new__(cls) -> H5Layout:
        """load layout"""
        return H5Layout(__this_dir__ / 'synpivimage.layout')

    @staticmethod
    def write():
        """write layout"""
        sntt = StandardNameTranslation()
        with layout.H5FileLayout(__this_dir__ / 'synpivimage.layout', 'w') as h5:
            ds = h5.create_dataset('images', shape=(1, 2, 2))
            ds.attrs['units'] = 'counts'
            ds.attrs['__ndim'] = 3
            ds.attrs['standard_name'] = sntt.translate('images')

            ds = h5.create_dataset('bit_depth', shape=(1,))
            ds.attrs['units'] = 'counts'
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('bit_depth')

            ds = h5.create_dataset('number_of_saturated_pixels', shape=(1,))
            ds.attrs['units'] = ''
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('number_of_saturated_pixels')

            ds = h5.create_dataset('particle_density', shape=(1,))
            ds.attrs['units'] = '1/pixel'
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('particle_density')

            ds = h5.create_dataset('particle_size_mean', shape=(1,))
            ds.attrs['units'] = 'pixel'
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('particle_size_mean')

            ds = h5.create_dataset('particle_size_std', shape=(1,))
            ds.attrs['units'] = 'pixel'
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('particle_size_std')

            ds = h5.create_dataset('laser_shape_factor', shape=(1,))
            ds.attrs['units'] = ''
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('laser_shape_factor')

            ds = h5.create_dataset('laser_width', shape=(1,))
            ds.attrs['units'] = 'm'
            ds.attrs['__ndim'] = 1
            ds.attrs['standard_name'] = sntt.translate('laser_width')

            h5.sdump()
