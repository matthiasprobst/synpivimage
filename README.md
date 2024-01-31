# synpivimage

Tool to build synthetic Particle Image Velocimetry (PIV) images

## Installation

Navigate to the repository folder and run

```bash
py -m pip install .
```

To use the package during development, install it accordingly:

```bash
py -m pip install -e .
```

To install `pytest` in order to run test optional dependencies must be installed:

```bash
py -m pip install -e ".[test]"
```

Then, execute `pytest` in the repository directory:

```bash
pytest
```

## Quick introduction

To generate synthetic particle images, configure a `ConfigManager`. It take the parameters and take care of generating
and writing the data to an `HDF5` file. For more explanation follow the
example `jupyter notebook` [here](./examples/generate_datasets.ipynb).

```python
import numpy as np

import synpivimage as spi

# load the default config dictionary:
cfg = spi.DEFAULT_CFG

# manipulated some parameters:
cfg['bit_depth'] = 8
cfg['nx'] = 31
cfg['ny'] = 31
cfg['sensor_gain'] = 0.6
cfg['particle_size_std'] = 1

# Set up the dictionary with variable ranges that will be varied:
image_size = cfg['nx'] * cfg['ny']
variation_dict = {'particle_number': np.arange(1, image_size * 0.1, 20).astype(int),
                  'particle_size_mean': (2, 3),
                  'laser_shape_factor': (1, 10)}

# init the config manager:
CFGs = spi.ConfigManager.from_variation_dict(initial_cfg=cfg,
                                             variation_dict=variation_dict,
                                             per_combination=3,
                                             shuffle=True)

# generate the images and store them in HDF5:
hdf_filename = CFGs.generate(data_directory='example_data_dir',
                             nproc=4, n_split=1000, overwrite=True)
```
