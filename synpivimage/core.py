"""Core module"""
import multiprocessing as mp
import numpy as np
import pathlib
import time

from .camera import Camera
from .laser import Laser
from .particles import Particles, compute_intensity_distribution

SQRT2 = np.sqrt(2)
# from .noise import add_camera_noise

__this_dir__ = pathlib.Path(__file__).parent

CPU_COUNT = mp.cpu_count()


def take_image(laser: Laser,
               cam: Camera,
               particles: Particles,
               particle_peak_count: int):
    """Takes an image of the particles

    1. Illuminates the particles (Note, that particles may lay outside the laser width! The
    function does not regenerate new particles!
    2. Captures the image
    3. Returns the image
    """
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
    intensity_factor = particle_peak_count / max_part_intensity / cam.qe / cam.sensitivity

    # compute the noise level:
    if cam.shot_noise:
        sqrtN = np.sqrt(cam.dark_noise)
    else:
        sqrtN = 0
    threshold_noise_level = cam.dark_noise + sqrtN

    # illuminate the particles (max intenstiy will be one. this is only the laser intensity assigned to the particles!)
    particles = laser.illuminate(particles, cam)
    # particles.intensity *= intensity_factor
    weakly_illuminated = particles.intensity * particle_peak_count < threshold_noise_level
    particles.mask = particles.mask & ~weakly_illuminated
    particles.intensity *= intensity_factor

    n_too_weak = np.sum(weakly_illuminated)
    print(f'Particles with intensity below the noise level: {n_too_weak}')

    n_valid = np.sum(particles.mask)
    n_total = len(particles)
    print(f'valid particles: {n_valid}:')
    print(f'valid particles: {n_valid / n_total * 100:.2f}%')
    print(f'total particles: {n_total}:')

    # capture the image
    print('Capturing the image...')
    st = time.time()
    img = cam.take_image(particles)
    et = time.time() - st
    print(f'...took: {et} s')

    n_total = len(particles)
    n_valid = np.sum(particles.mask)
    print(f'valid particles: {n_valid}:')
    print(f'ppp={n_valid / (cam.ny * cam.nx):.4f}')

    return img
