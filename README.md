# synpivimage

Tool to build synthetic Particle Image Velocimetry (PIV) images based on commonly accepted literature.

It uses the commonly used assumptions and DOES NOT include optical aberrations, such as lens distortion or such. Noise
can be added and particles can be moved, to create a synthetic A-B-image pair.

A GUI helps to investigate the effect of the parameters on the image(s).

**Note, that this package is still under development and the API might change.**

The image particles density distribution is modelled as described in the Book "Particle Image Velocimetry: A Practical
Guide" by Raffel et al. (https://doi.org/10.1007/978-3-319-68852-7).

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

### Simple A-B-image generation:

```python


import synpivimage as spi

# load the default config dictionary:
cfg = spi.get_default()

# manipulated some parameters:
cfg.bit_depth = 8
cfg.nx = 31
cfg.ny = 31
cfg.particle_size_std = 1

# Set up the dictionary with variable ranges that will be varied:
image_size = cfg.nx * cfg.ny

from synpivimage import generate_image

imgA, attrsA, part_infoA = generate_image(
    cfg,
    particle_data=None
)

# displace the particles (here a random displacement)
# We need to use the special class to displace the particles as it will 
# take care of new particles moving into the laser light sheet
from synpivimage import velocityfield

cfield = velocityfield.ConstantField(dx=2.3, dy=1.6, dz=0)
displaced_particle_data = cfield.displace(cfg=cfg, part_info=part_infoA)

imgB, attrsB, part_infoB = generate_image(
    cfg,
    particle_data=displaced_particle_data
)

from synpivimage import io

io.imwrite8('img8_A.tif', imgA)
io.imwrite8('img8_B.tif', imgB)
```

### Varying multiple parameters at once:

```python

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

