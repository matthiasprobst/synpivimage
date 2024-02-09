"""Core module"""
import multiprocessing as mp
import numpy as np
import pathlib
import time
from pivimage import PIVImage

from .camera import Camera
from .laser import Laser
from .log import DEFAULT_LOGGER
from .particles import Particles, compute_intensity_distribution

SQRT2 = np.sqrt(2)
# from .noise import add_camera_noise

__this_dir__ = pathlib.Path(__file__).parent

CPU_COUNT = mp.cpu_count()


def take_image(laser: Laser,
               cam: Camera,
               particles: Particles,
               particle_peak_count: int,
               **kwargs) -> PIVImage:
    """Takes an image of the particles

    1. Illuminates the particles (Note, that particles may lay outside the laser width! The
    function does not regenerate new particles!
    2. Captures the image
    3. Returns the image
    """
    logger = kwargs.get('logger', DEFAULT_LOGGER)
    DEBUG_LEVEL = kwargs.get('debug_level', 0)
    # compute the particle intensity factor in order to reach particle_peak_count
    # For this, call the error function
    mean_particle_size = np.mean(particles.size)
    max_part_intensity = compute_intensity_distribution(
        x=0,
        y=0,
        xp=0,
        yp=0,
        dp=mean_particle_size,
        sigmax=cam.sigmax,
        sigmay=cam.sigmay,
        fill_ratio_x=cam.fill_ratio_x,
        fill_ratio_y=cam.fill_ratio_y
    )
    intensity_factor = (particle_peak_count + 1) / max_part_intensity / cam.qe / cam.sensitivity

    # compute the noise level:
    if cam.shot_noise:
        sqrtN = np.sqrt(cam.dark_noise)
    else:
        sqrtN = 0
    threshold_noise_level = cam.dark_noise + sqrtN

    # illuminate the particles (max intensity will be one. this is only the laser intensity assigned to the particles!)
    particles = laser.illuminate(particles, debug_level=DEBUG_LEVEL)

    xmask = np.logical_and(0 < particles.x, particles.x < cam.nx - 1)
    ymask = np.logical_and(0 < particles.y, particles.y < cam.ny - 1)
    particles.mask &= xmask & ymask

    # particles.intensity *= intensity_factor
    weakly_illuminated = particles.intensity * particle_peak_count < threshold_noise_level
    particles.mask &= ~weakly_illuminated
    particles.intensity = np.multiply(particles.intensity, intensity_factor)

    n_too_weak = np.sum(weakly_illuminated)
    logger.debug(f'Particles with intensity below the noise level: {n_too_weak}')

    n_valid = np.sum(particles.mask)
    n_total = len(particles)
    logger.debug(f'valid particles: {n_valid}:')
    logger.debug(f'valid particles: {n_valid / n_total * 100:.2f}%')
    logger.debug(f'total particles: {n_total}:')

    # capture the image
    logger.debug('Capturing the image...')
    st = time.time()
    img, n_saturated = cam.take_image(particles)
    et = time.time() - st
    logger.debug(f'...took: {et} s')

    n_valid = np.sum(particles.mask)
    logger.debug(f'valid particles: {n_valid}:')
    logger.debug(f'ppp={n_valid / (cam.ny * cam.nx):.4f}')

    return PIVImage.from_array(np.asarray(img))
