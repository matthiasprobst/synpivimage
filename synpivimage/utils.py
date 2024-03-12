import logging
from typing import Tuple

import numpy as np

from .camera import Camera
from .core import take_image
from .laser import Laser
from .particles import Particles

logger = logging.getLogger('synpivimage')


def generate_particles(ppp: float,
                       *,
                       dx_max: Tuple[float, float],
                       dy_max: Tuple[float, float],
                       dz_max: Tuple[float, float],
                       camera: Camera,
                       laser: Laser,
                       iter_max: int = 40):
    """Generates a particle class based on the current setup and given max displacements

    Parameters
    ----------
    ppp: float
        Target particles per pixel (ppp). Value must be between 0 and 1
    dx_max: Tuple[float, float]
        Maximal displacement in x-direction [lower, upper]
    dy_max: Tuple[float, float]
        Maximal displacement in y-direction [lower, upper]
    dz_max: Tuple[float, float]
        Maximal displacement in z-direction [lower, upper]
    """
    logger.debug(f'Generating particles with a ppp of {ppp}')
    assert 0 < ppp < 1, f"Expected ppp to be between 0 and 1, got {ppp}"

    if laser.shape_factor > 100:
        zmin = -laser.width / 2 - 0.01 * laser.width
        zmax = laser.width / 2 - + 0.01 * laser.width
    else:
        zmin = -laser.width
        zmax = laser.width
    area = (camera.nx + (dx_max[1] - dx_max[0])) * (camera.ny + (dy_max[1] - dy_max[0]))
    N = int(area * ppp)
    # Ntarget = ppp * camera.size

    curr_ppp = 0
    # rel_dev = abs((curr_ppp - ppp) / ppp)

    i = 0
    while abs((curr_ppp - ppp) / ppp) > 0.01:
        i += 1
        if i == 1:
            # generate a initial random distribution. From this. particles are either added or removed
            logger.debug(f'Generate initial random distribution of {N} particles')
            xe = np.random.uniform(min(-dx_max[1], 0), max(camera.nx, camera.nx - dx_max[0]), N)
            ye = np.random.uniform(min(-dy_max[1], 0), max(camera.ny, camera.ny - dy_max[0]), N)
            ze = np.random.uniform(min(zmin, zmin - dz_max[1]), max(zmax, zmax + dz_max[0]), N)

        particles = Particles(
            x=xe,
            y=ye,
            z=ze,
            size=np.ones_like(xe) * 2
        )
        _img, _part = take_image(particles=particles,
                                 camera=camera,
                                 laser=laser,
                                 particle_peak_count=1000)
        curr_ppp = _part.active.sum() / camera.size
        # print(f'curr ppp: {curr_ppp:.5f}')
        diff_ppp = ppp - curr_ppp
        # print(f'diff ppp: {diff_ppp:.5f}')
        Nadd = int(diff_ppp * camera.size)  # dampen the addition by 10%

        if Nadd == 0:
            logger.debug('Stopping early because no new particles to be added.')
            break

        logger.debug(f'Generate particles. Iteration {i}/{iter_max}: curr ppp: {curr_ppp:.5f}. '
                     f'diff ppp: {diff_ppp:.5f}. Adding {Nadd} particles')

        if Nadd > 0:
            # generate new particles within the given range
            xe_new = np.random.uniform(min(-dx_max[1], 0), max(camera.nx, camera.nx - dx_max[0]), Nadd)
            ye_new = np.random.uniform(min(-dy_max[1], 0), max(camera.ny, camera.ny - dy_max[0]), Nadd)
            ze_new = np.random.uniform(min(zmin, zmin - dz_max[1]), max(zmax, zmax + dz_max[0]), Nadd)
            # add them to the existing particles
            _n_new = len(xe) + Nadd
            xe = np.concatenate([xe, xe_new])
            ye = np.concatenate([ye, ye_new])
            ze = np.concatenate([ze, ze_new])
            assert len(xe) == N + Nadd
        elif Nadd < 0:
            # remove particles
            idx = np.random.choice(N, -Nadd, replace=False)
            xe = np.delete(xe, idx)
            ye = np.delete(ye, idx)
            ze = np.delete(ze, idx)
            assert len(xe) == N + Nadd

        N += Nadd
        err = abs((curr_ppp - ppp) / ppp)

        if err < 0.01:
            logger.debug(f'Convergence crit (err < 0.01- reached. Residual error {err * 100:.1f} %')

        if i >= iter_max:
            logger.debug(f'Reached max iteration of {iter_max}')
            break

    return _part
