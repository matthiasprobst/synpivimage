"""Layout created with h5RDMtoolbox"""

from h5rdmtoolbox.conventions import StandardNameTableTranslation
from h5rdmtoolbox.conventions import layout

sntt = StandardNameTableTranslation.from_yaml('synpivimage-to-synpiv-v1.yml')

with layout.H5FileLayout('synpivimage.layout', 'w') as h5:

    ds = h5.create_dataset('images', shape=(1, 2, 2))
    ds.attrs['units'] = 'counts'
    ds.attrs['__ndim'] = 3
    ds.attrs['standard_name'] = sntt.translate('images')

    ds = h5.create_dataset('bit_depth', shape=(1,))
    ds.attrs['units'] = ''
    ds.attrs['__ndim'] = 1
    ds.attrs['standard_name'] = sntt.translate('bit_depth')

    ds = h5.create_dataset('number_of_saturated_pixels', shape=(1,))
    ds.attrs['units'] = ''
    ds.attrs['__ndim'] = 1
    ds.attrs['standard_name'] = sntt.translate('number_of_saturated_pixels')

    ds = h5.create_dataset('particle_density', shape=(1,))
    ds.attrs['units'] = '1/px'
    ds.attrs['__ndim'] = 1
    ds.attrs['standard_name'] = sntt.translate('particle_density')

    ds = h5.create_dataset('particle_size_mean', shape=(1,))
    ds.attrs['units'] = 'px'
    ds.attrs['__ndim'] = 1
    ds.attrs['standard_name'] = sntt.translate('particle_size_mean')

    ds = h5.create_dataset('particle_size_std', shape=(1,))
    ds.attrs['units'] = 'px'
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
