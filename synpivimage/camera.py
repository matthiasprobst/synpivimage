import numpy as np
from pydantic import BaseModel
from typing import Tuple
from typing_extensions import Annotated

from . import noise
from .component import Component
from .particles import Particles, model_image_particles
from .validation import PositiveInt, PositiveFloat, ValueRange

Efficiency = Annotated[float, ValueRange(0, 1)]
FillRatio = Annotated[float, ValueRange(0, 1)]


class Camera(BaseModel, Component):
    """Camera Model"""
    nx: PositiveInt
    ny: PositiveInt
    bit_depth: PositiveInt
    qe: Efficiency
    sensitivity: Efficiency
    baseline_noise: float
    dark_noise: float
    shot_noise: float
    fill_ratio_x: FillRatio
    fill_ratio_y: FillRatio
    particle_image_diameter: PositiveFloat

    # sigmax: PositiveFloat
    # sigmay: PositiveFloat

    @property
    def size(self) -> int:
        """Size of the sensor in pixels"""
        return int(self.nx * self.ny)

    @property
    def max_count(self):
        """Max count of the sensor"""
        return int(2 ** self.bit_depth - 1)

    def _quantize(self, electrons) -> Tuple[np.ndarray, int]:
        """Quantize the electrons to the bit depth"""
        max_adu = self.max_count
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

        return np.asarray(adu), int(n_saturated_pixels)

    def _capture(self, irrad_photons):
        """Capture the image and add noise"""
        electrons = noise.add_noise(irrad_photons,
                                    self.shot_noise,
                                    self.baseline_noise,
                                    self.dark_noise,
                                    self.qe)
        return electrons

    def take_image(self, particles: Particles) -> Tuple[np.ndarray, int]:
        """capture and quantize the image.

        .. note::
            The definition of the image particle diameter is the diameter of the
            particle image in pixels, where the normalized gaussian is equal to $e^{-2}$,
            which is a full width of $4 \sigma$.

        Returns image and number of saturated pixels.
        """
        irrad_photons = model_image_particles(
            particles[particles.active],
            nx=self.nx,
            ny=self.ny,
            sigmax=self.particle_image_diameter / 4,
            sigmay=self.particle_image_diameter / 4,
            fill_ratio_x=self.fill_ratio_x,
            fill_ratio_y=self.fill_ratio_y
        )
        electrons = self._capture(irrad_photons)
        return self._quantize(electrons)
