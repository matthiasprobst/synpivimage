"""Core module"""
import logging
import multiprocessing as mp
import numpy as np
import pathlib
import time
from pivimage import PIVImage
from typing import Tuple

from .camera import Camera
from .laser import Laser
from .particles import Particles, compute_intensity_distribution, ParticleFlag

LOGGER = logging.getLogger('synpivimage')
COUNT_EDGE_PARTICLES = False
NSIGMA_NOISE_THRESHOLD = 4  # 4*dark_noise=4*sigma_dark_noise should be the threshold for the particle intensity.
# if the peak intensity is below that, the particle is considered not be have an influence on the cross correlation.
SQRT2 = np.sqrt(2)
# from .noise import add_camera_noise

__this_dir__ = pathlib.Path(__file__).parent

CPU_COUNT = mp.cpu_count()


def take_image(laser: Laser,
               cam: Camera,
               particles: Particles,
               particle_peak_count: int,
               **kwargs) -> Tuple[PIVImage, Particles]:
    """Takes an image of the particles

    1. Illuminates the particles (Note, that particles may lay outside the laser width! The
    function does not regenerate new particles!
    2. Captures the image
    3. Returns the image

    Parameters
    ----------
    laser : Laser
        The laser object containing the laser parameters
    cam : Camera
        The camera object containing the camera parameters
    particles : Particles
        The particles to be imaged
    particle_peak_count : int
        The peak count of the particles
    kwargs : dict
        Additional parameters
    """
    logger = kwargs.get('logger', LOGGER)
    # compute the particle intensity factor in order to reach particle_peak_count
    # For this, call the error function
    mean_particle_size = np.mean(particles.size)
    max_part_intensity = compute_intensity_distribution(
        x=0,
        y=0,
        xp=0,
        yp=0,
        dp=mean_particle_size,
        sigmax=cam.particle_image_diameter / 4,
        sigmay=cam.particle_image_diameter / 4,
        fill_ratio_x=cam.fill_ratio_x,
        fill_ratio_y=cam.fill_ratio_y
    )
    intensity_factor = (particle_peak_count + 1) / max_part_intensity / cam.qe / cam.sensitivity
    # assert int(intensity_factor * max_part_intensity) == 1000

    # compute the noise level:
    if cam.shot_noise:
        sqrtN = np.sqrt(max_part_intensity * intensity_factor)
    else:
        sqrtN = 0

    # illuminate the particles (max intensity will be one. this is only the laser intensity assigned to the particles!)
    particles = laser.illuminate(particles)  # range between 0 and 1
    assert np.min(particles.source_intensity) >= 0
    assert np.max(particles.source_intensity) <= 1

    hips = cam.particle_image_diameter / 2  # half image particle size
    if COUNT_EDGE_PARTICLES:
        """An edge particle has its center at the border of the image"""
        xflag = np.logical_and(-hips < particles.x, particles.x + hips < cam.nx - 1)
        yflag = np.logical_and(-hips < particles.y, particles.y + hips < cam.ny - 1)
    else:
        # particles with half their size away from the border are considered
        xflag = np.logical_and(hips < particles.x, particles.x + hips < cam.nx - 1)
        yflag = np.logical_and(hips < particles.y, particles.y + hips < cam.ny - 1)

    # these particles are active/illuminated
    in_fov = xflag & yflag
    particles.flag[~in_fov] = ParticleFlag.DISABLED.value
    particles.flag[in_fov] = ParticleFlag.IN_FOV.value  # in the next step we check if they are illuminated

    # the dark noise should not be greater that the particle intensity, otherwise the particle will not be visible
    # the total particle intensity is the baseline noise + the particle intensity + the shot noise (if enabled)
    # + the dark noise.
    illumination_threshold = max(NSIGMA_NOISE_THRESHOLD * cam.dark_noise + sqrtN, np.exp(-2) * particle_peak_count)
    weakly_illuminated = particles.source_intensity * particle_peak_count <= illumination_threshold
    # disable the particles due to weak illumination (mark only IN-FOV-particles like this!):
    particles.flag[weakly_illuminated] += ParticleFlag.OUT_OF_PLANE.value
    particles.flag[~weakly_illuminated] += ParticleFlag.ILLUMINATED.value
    # update the particle source intensities
    particles.source_intensity = np.multiply(particles.source_intensity, intensity_factor)
    # particles.image_max_noiseless_intensity = np.multiply(particles.source_intensity, intensity_factor)

    n_too_weak = np.sum(weakly_illuminated)
    logger.debug(f'Particles with intensity below the noise level: {n_too_weak}')

    logger.debug('=== STATISTICS ===')
    n_relevant = np.asarray(particles.flag & (ParticleFlag.IN_FOV.value + ParticleFlag.ILLUMINATED.value),
                            dtype=bool).sum()
    n_total = len(particles)
    logger.debug(f'total particles: {len(particles.flag)}:')
    logger.debug(f'FOV and illuminated particles: {n_relevant} ({n_relevant / n_total * 100:.2f}%)')
    flag = (ParticleFlag.IN_FOV.value | ParticleFlag.OUT_OF_PLANE.value)
    n_out_of_plane = np.sum(np.asarray(particles.flag & flag, dtype=bool) == 6)
    logger.debug(f'Out Of Plane in FOV: {n_out_of_plane}:')

    # capture the image
    logger.debug('Capturing the image...')
    st = time.time()
    img, n_saturated = cam.take_image(particles)
    et = time.time() - st
    logger.debug(f'...took: {et} s')

    n_valid = np.sum(particles.flag)
    logger.debug(f'valid particles: {n_valid}:')
    logger.debug(f'ppp={n_valid / (cam.ny * cam.nx):.4f}')

    return PIVImage.from_array(np.asarray(img)), particles
