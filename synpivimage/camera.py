import numpy as np
from typing import Tuple

from . import noise
from .particles import Particles, model_image_particles


class ADU(np.ndarray):

    def __new__(cls, input_array, n_saturated_pixels: int):
        obj = np.asarray(input_array).view(cls)
        obj.n_saturated_pixels = n_saturated_pixels
        return obj


class Camera:

    def __init__(self, *,
                 nx,
                 ny,
                 bit_depth,
                 qe,
                 sensitivity,
                 baseline_noise,
                 dark_noise,
                 shot_noise,
                 fill_ratio_x: float,
                 fill_ratio_y: float,
                 sigmax,
                 sigmay):
        self.nx = nx
        self.ny = ny
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.fill_ratio_x = fill_ratio_x
        self.fill_ratio_y = fill_ratio_y
        self.bit_depth = bit_depth
        self.qe = qe
        self.sensitivity = sensitivity
        self.baseline_noise = baseline_noise
        self.dark_noise = dark_noise
        self.shot_noise = shot_noise

    def _quantize(self, electrons) -> Tuple[np.ndarray, int]:
        """Quantize the electrons to the bit depth"""
        max_adu = int(2 ** self.bit_depth - 1)
        adu = electrons * self.sensitivity
        _saturated_pixels = adu > max_adu
        n_saturated_pixels = np.sum(_saturated_pixels)
        adu[adu > max_adu] = max_adu  # model saturation
        if self.bit_depth == 8:
            adu = adu.astype(np.uint8)
        elif self.bit_depth == 16:
            adu = adu.astype(np.uint16)
        else:
            raise ValueError(f"Bit depth {self.bit_depth} not supported")

        return ADU(adu, n_saturated_pixels)

    def _capture(self, irrad_photons):
        """Capture the image and add noise"""
        electrons = noise.add_noise(irrad_photons, self.shot_noise, self.baseline_noise, self.dark_noise, self.qe)
        return electrons

    def take_image(self,
                   particles: Particles):
        """capture and quantize the image"""
        irrad_photons = model_image_particles(
            particles[particles.mask],
            nx=self.nx,
            ny=self.ny,
            sigmax=self.sigmax,
            sigmay=self.sigmay,
            fill_ratio_x=self.fill_ratio_x,
            fill_ratio_y=self.fill_ratio_y
        )
        electrons = self._capture(irrad_photons)
        return self._quantize(electrons)
