# synpivimage - A transparent way of generating synthetic PIV images

![Tests](https://github.com/matthiasprobst/synpivimage/actions/workflows/tests.yml/badge.svg)
![DOCS](https://codecov.io/gh/matthiasprobst/synpivimage/branch/dev/graph/badge.svg)
![pyvers](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)

This tool let's you generate synthetic Particle Image Velocimetry (PIV) images based on methods described in
literature (mainly based on "Particle Image Velocimetry: A Practical Guide" by Raffel et
al. (https://doi.org/10.1007/978-3-319-68852-7)).

## Highlights

- The user has full control over the parameters.
- Data and metadata can be stored in a single HDF5 file or classically in TIF files.
- The metadata (camera and laser settings) can be stored in a separate JSON-LD file, which adheres to the state of the
  art way of storing metadata, allowing for easy integration into any other software or database.

## Installation

Navigate to the repository folder and run

```bash
pip install synpivimage
```

## Documentation

A comprehensive documentation can be found [here](https://synpivimage.readthedocs.io/en/latest/).

### Minimal example:

```python
import numpy as np

import synpivimage

cam = synpivimage.Camera(
    nx=256,
    ny=256,
    bit_depth=16,
    qe=1,
    sensitivity=1,
    baseline_noise=50,
    dark_noise=10,
    shot_noise=False,
    fill_ratio_x=1.0,
    fill_ratio_y=1.0,
    particle_image_diameter=4  # px
)

laser = synpivimage.Laser(
    width=0.25,
    shape_factor=2
)

n = 100
particles = synpivimage.Particles(
    x=np.random.uniform(-3, cam.nx - 1, n),
    y=np.random.uniform(-4, cam.ny - 1, n),
    z=np.zeros(n),
    size=np.ones(n) * 2,
)

imgA, partA = synpivimage.take_image(laser,
                                     cam,
                                     particles,
                                     particle_peak_count=1000)

displaced_particles = partA.displace(dx=2.1, dy=3.4)

imgB, partB = synpivimage.take_image(laser,
                                     cam,
                                     displaced_particles,
                                     particle_peak_count=1000)

with synpivimage.Imwriter(case_name="test_case",
                          camera=cam,
                          laser=laser) as iw:
    iw.writeA(0, imgA, partA)
    iw.writeB(0, imgB, partB)

with synpivimage.HDF5Writer(case_name="test_case",
                            n_images=1,
                            camera=cam,
                            laser=laser) as hw:
    hw.writeA(0, imgA, partA)
    hw.writeB(0, imgB, partB)
```

## Developers

### Testing

Call the following inside the package directory to run the tests (with coverage)

```bash
pytest --cov=synpivimage --cov-report html
```

### Contributing

Contributions are welcome! Please open an issue or a pull request.

## License

This project is licensed under the MIT License.