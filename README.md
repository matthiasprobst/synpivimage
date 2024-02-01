# synpivimage

Tool to build synthetic Particle Image Velocimetry (PIV) images.

The image particles density distribution is modelled as described in the Book "Particle Image Velocimetry: A Practical Guide" by 
Raffel et al. (https://doi.org/10.1007/978-3-319-68852-7).

The effective sensor count is computed as follows:

counts = (RLI x 2**BIT + NOISE ) x QE

- RLI: relative laser intensity; value range: (0 , 1]
- BIT: bit depth of camera, e.g. 8 or 16 (RLI x 2**BIT gives the maximum number of photons emitted by a particle)
- NOISE: noise (in number of photons)
- QE: quantum efficiency (conversion efficiency of photons to electrons)
- counts: number of electrons = counts on the sensor = image intensity

For *no noise* and RLI=1 a particle will be seen with maximal pixel count in the image. This means, that a slight 
overlap will saturate the camera. Therefore the VLI should be < 1
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
