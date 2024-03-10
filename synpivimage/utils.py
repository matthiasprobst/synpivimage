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
                       iter_max: int = 20):
    """Generates a particle class based on the current setup and given max displacements"""
    logger.debug('generating particles with a certain ppp of {ppp}')
    assert 0 < ppp < 1, f"Expected ppp to be between 0 and 1, got {ppp}"

    i = 0
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

    while abs((curr_ppp - ppp) / ppp) > 0.01:
        i += 1
        logger.debug(f'--- Iteration {i} ----- ')
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
        Nadd = int(diff_ppp * camera.size)

        if Nadd == 0:
            logger.debug(' > Stopping early because no new particles to be added')
            break

        logger.debug('curr ppp  |  diff ppp ')
        logger.debug(f' {curr_ppp:.5f}  |  {diff_ppp:.5f}   --> adding {Nadd} particles')
        N += Nadd
        err = abs((curr_ppp - ppp) / ppp)
        # print(f'rel diff in ppp: {rel_diff:.5f}')

        if err < 0.01:
            logger.debug(f' Convergence crit reached')
            logger.debug(f' > Residual error {err * 100:.1f} %')

        if i > iter_max:
            logger.debug(f' > Reached max iteration of {iter_max}')
            break

    return _part
