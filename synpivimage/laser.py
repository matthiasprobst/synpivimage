import matplotlib.pyplot as plt
import numpy as np

from .camera import Camera
from .particles import Particles

SQRT2 = np.sqrt(2)

DEBUG_LEVEL = 2  # 2 will plot images


class tophat:
    """Tophat function"""

    def __init__(self, dz0):
        self.dz0 = dz0

    def __call__(self, z):
        intenstiy = np.ones_like(z)
        intenstiy[z < -self.dz0] = 0
        intenstiy[z > self.dz0] = 0
        return intenstiy


class Laser:
    """Laser class. This class will be used to illuminate the particles"""

    def __init__(self,
                 shape_factor: int,
                 width: float):
        self.shape_factor = shape_factor
        self.width = width  # width of the laser, not the effective laser width

    def get_effective_laser_width(self, cam: Camera):
        """The effective laser width is the width where a particle
        can be (theoretically) be distinguished from the background.
        For this to compute, we need to know the laser properties and
        the camera properties. The effective laser width is the width
        where the particle intensity exceeds the noise level
        """

    def illuminate(self,
                   particles: Particles,
                   cam: Camera):
        dz0 = SQRT2 * self.width / 2
        s = self.shape_factor
        if s == 0:
            laser_intensity = lambda z: 1
        elif s > 100:
            laser_intensity = tophat(dz0)
        else:
            laser_intensity = lambda z: np.exp(-1 / np.sqrt(2 * np.pi) * np.abs(2 * z ** 2 / dz0 ** 2) ** s)
        particles.intensity = laser_intensity(particles.z)

        inside_laser = particles.intensity > np.exp(-2)
        particles.mask = inside_laser  # mask for the particles inside the laser

        if DEBUG_LEVEL > 0:
            n_removed = np.sum(~inside_laser)
            n_total = len(particles)
            perc_removed = n_removed / n_total * 100
            print(f'Removed {n_removed} ({perc_removed} %) particles because they are outside the laser,'
                  f' which is defined as an intensity below exp(-2)')

        if DEBUG_LEVEL > 1:
            plt.figure()
            plt.plot(particles.z[inside_laser], particles.intensity[inside_laser], 'o', color='g')
            plt.plot(particles.z[~inside_laser], particles.intensity[~inside_laser], 'o', color='r')
            plt.xlabel('z / real arbitrary units')
            plt.ylabel('Normalized particle intensity in beam / -')
            plt.grid()
            plt.show()
        return particles

    def illuminate12(self,
                     mean_size: float,
                     std_size: float,
                     n_particles: int,
                     cam: Camera):
        """First shot. No particle input expected. The
        particles will be generated randomly and uniformly."""
        if std_size == 0:
            particle_distribution_props = mean_size
        else:
            assert std_size > 0
            particle_distribution_props = (mean_size, std_size)
        particles = Particles.generate_uniform(n_particles,
                                               particle_distribution_props,
                                               cam.x_bounds,
                                               cam.y_bounds,
                                               cam.z_bounds)
