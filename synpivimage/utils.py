import logging
import warnings
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
                       size: float,
                       camera: Camera,
                       laser: Laser,
                       **kwargs) -> Particles:
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
    camera: Camera
        camera
    laser: Laser
        laser
    size: float
        Particle size
    kwargs:
        iter_max: int=40
            Max. iteration for particle number determination algorithm
        N_max: int=10**7
            Max. number of particles

    Returns
    -------
    particles: Particles
        The generated particles
    """

    iter_max = kwargs.get('iter_max', 40)
    N_max = kwargs.get('N_ma', 10 ** 7)

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
    if N < 2:
        raise ValueError(
            'Number of particles is too low. For generation of only one particle,'
            ' please use discrete generation by providing the coordinates'
        )
    logger.debug(f'Initial particle count: {N}')
    # Ntarget = ppp * camera.size

    curr_ppp = 0
    # rel_dev = abs((curr_ppp - ppp) / ppp)

    _xlim = min(-dx_max[1], 0), max(camera.nx, camera.nx - dx_max[0])
    _ylim = min(-dy_max[1], 0), max(camera.ny, camera.ny - dy_max[0])
    _zlim = min(zmin, zmin - dz_max[1]), max(zmax, zmax + dz_max[0])

    x_extent = _xlim[1] - _xlim[0]
    y_extent = _ylim[1] - _ylim[0]

    n_too_much_noise = 0
    i = 0
    while abs((curr_ppp - ppp) / ppp) > 0.01:
        i += 1
        if i == 1:
            # generate a initial random distribution. From this. particles are either added or removed
            logger.debug(f'Generate initial random distribution of {N} particles')
            xe = np.random.uniform(*_xlim, N)
            ye = np.random.uniform(*_ylim, N)
            ze = np.random.uniform(*_zlim, N)

        particles = Particles(
            x=xe,
            y=ye,
            z=ze,
            size=size * np.ones_like(xe)
        )
        _img, _part = take_image(particles=particles,
                                 camera=camera,
                                 laser=laser,
                                 particle_peak_count=1000)
        curr_ppp = _part.get_ppp(camera.size)  # _part.active.sum() / camera.size

        diff_ppp = ppp - curr_ppp

        Nadd = int(diff_ppp * x_extent * y_extent)

        if Nadd == 0:
            logger.debug('Stopping early because no new particles to be added.')
            break

        logger.debug(f'Generate particles. Iteration {i}/{iter_max}: curr ppp: {curr_ppp:.5f}. '
                     f'diff ppp: {diff_ppp:.5f}. Adding {Nadd} particles')
        if Nadd > 0:
            # generate new particles within the given range
            xe_new = np.random.uniform(*_xlim, Nadd)
            ye_new = np.random.uniform(*_ylim, Nadd)
            ze_new = np.random.uniform(*_zlim, Nadd)
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
        
        if N > N_max:
            raise ValueError(f'Number of particles exceeded maximum of {N_max}. '
                             f'Consider increasing the number of iterations or the particle size')

        if curr_ppp == 0 and _part.in_fov.sum() > 0:
            n_too_much_noise += 1
            warnings.warn('No particles in the field of view are illuminated beyond the laser noise. Consider using a'
                          'smaller noise level or a larger particle size.')
            if n_too_much_noise > 10:
                raise ValueError('Too many iterations without any particles in the field of view. Stopping.')

        err = abs((curr_ppp - ppp) / ppp)

        if err < 0.01:
            logger.debug(f'Convergence crit (err < 0.01- reached. Residual error {err * 100:.1f} %')

        if i >= iter_max:
            logger.debug(f'Reached max iteration of {iter_max}')
            break

    _part._xlim = _xlim
    _part._ylim = _ylim
    _part._zlim = _zlim
    return _part
