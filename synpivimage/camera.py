import numpy as np
import pathlib
from pydantic import BaseModel
from typing import Tuple, Union
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

    @property
    def size(self) -> int:
        """Size of the sensor in pixels"""
        return int(self.nx * self.ny)

    @property
    def max_count(self):
        """Max count of the sensor"""
        return int(2 ** self.bit_depth - 1)

    def _quantize(self, electrons) -> Tuple[np.ndarray, int]:
        """Quantize the electrons to the bit depth

        Parameters
        ----------
        electrons : np.ndarray
            The number of electrons

        Returns
        -------
        np.ndarray
            The quantized image
        int
            The number of saturated pixels
        """
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
        # active = particles.active
        active = particles.in_fov
        irrad_photons, particles.max_image_photons[active] = model_image_particles(
            particles[active],
            nx=self.nx,
            ny=self.ny,
            sigmax=self.particle_image_diameter / 4,
            sigmay=self.particle_image_diameter / 4,
            fill_ratio_x=self.fill_ratio_x,
            fill_ratio_y=self.fill_ratio_y
        )
        electrons = self._capture(irrad_photons)
        particles.image_electrons[active] = self._capture(particles.max_image_photons[active])
        particles.image_quantized_electrons[active] = self._quantize(particles.image_electrons[active])[0]
        return self._quantize(electrons)

    def save_jsonld(self, filename: Union[str, pathlib.Path]):
        """Save the component to JSON"""
        from pivmetalib import pivmeta
        from pivmetalib import m4i
        filename = pathlib.Path(filename)  # .with_suffix('.jsonld')

        def _build_iri(sn: str):
            if sn:
                if sn.startswith('http'):
                    return sn
                return f"https://matthiasprobst.github.io/pivmeta#{sn}"
            return None

        def _build_variable(value, standard_name=None, unit=None, qkind=None, label=None, description=None):
            kwargs = {'hasNumericalValue': value}
            if label:
                kwargs['label'] = label
            if standard_name:
                kwargs['hasStandardName'] = _build_iri(standard_name)
            if unit:
                kwargs['hasUnit'] = unit
            if qkind:
                kwargs['hasKindOfQuantity'] = qkind
            if description:
                kwargs['hasVariableDescription'] = description
            return pivmeta.NumericalVariable(
                **kwargs
            )

        sn_dict = {
            'nx': 'sensor_pixel_width',
            'ny': 'sensor_pixel_height',
            'bit_depth': '',
            'fill_ratio_x': 'sensor_pixel_width_fill_factor',
            'fill_ratio_y': 'sensor_pixel_height_fill_factor',
            'particle_image_diameter': 'image_particle_diameter'
        }
        descr_dict = {
            'qe': 'quantum efficiency',
            'dark_noise': 'Dark noise is the standard deviation of a gaussian noise model',
            'baseline_noise': 'Dark noise is the mean value of a gaussian noise model'
        }
        unit_dict = {
            'nx': '',
            'ny': '',
            'bit_depth': 'http://qudt.org/vocab/unit/BIT',
        }
        qkind_dict = {
            'nx': "https://qudt.org/schema/qudt/DimensionlessUnit",
            'ny': "https://qudt.org/schema/qudt/DimensionlessUnit",
            'bit_depth': 'http://qudt.org/schema/qudt/CountingUnit'
        }

        hasParameter = []
        field_dict = self.model_dump(exclude_none=True)
        shot_noise = field_dict.pop('shot_noise')
        for key, value in field_dict.items():
            hasParameter.append(
                _build_variable(
                    label=key,
                    value=value,
                    unit=unit_dict.get(key, None),
                    qkind=qkind_dict.get(key, None),
                    standard_name=sn_dict.get(key, None),
                    description=descr_dict.get(key, None)
                )
            )
        shot_noise_txt_value = 'true' if shot_noise else 'false'
        hasParameter.append(
            m4i.variable.TextVariable(label='shot_noise',
                                      hasStringValue=shot_noise_txt_value)
        )

        camera = pivmeta.DigitalCamera(
            hasParameter=hasParameter
        )
        with open(filename, 'w') as f:
            f.write(
                camera.dump_jsonld()
            )
        return filename
