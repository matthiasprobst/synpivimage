import pathlib
from typing import Tuple, Union, Optional

import numpy as np
from ontolutils.namespacelib import QUDT_UNIT, QUDT_KIND
from pivmetalib import PIVMETA
from pivmetalib.m4i import NumericalVariable
from pydantic import BaseModel
from ssnolib.ssno import StandardName
from typing_extensions import Annotated

from . import noise
from .component import Component, load_jsonld
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
    shot_noise: bool
    fill_ratio_x: FillRatio
    fill_ratio_y: FillRatio
    particle_image_diameter: PositiveFloat
    seed: Optional[int] = None

    @property
    def size(self) -> int:
        """Size of the sensor in pixels (nx x ny)"""
        return int(self.nx * self.ny)

    @property
    def max_count(self):
        """Max count of the sensor, which is computed from the
        bit depth `b` of the sensor.

        .. math

            c_{max} = 2**b -1
        """
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
                                    self.qe,
                                    rs=np.random.RandomState(self.seed)
                                    )
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

    def model_dump_jsonld(self) -> str:
        """Return JSON-LD str"""
        from pivmetalib import pivmeta
        from pivmetalib import m4i
        from .codemeta import get_package_meta

        def _build_variable(value, standard_name=None, unit=None, qkind=None, label=None, description=None):
            kwargs = {'hasNumericalValue': value}
            if label:
                kwargs['label'] = label
            if standard_name:
                kwargs['hasStandardName'] = standard_name
            if unit:
                kwargs['hasUnit'] = unit
            if qkind:
                kwargs['hasKindOfQuantity'] = qkind
            if description:
                kwargs['hasVariableDescription'] = description
            return NumericalVariable(
                **kwargs
            )

        sn_dict = {
            'nx': StandardName(standardName='sensor_pixel_width', unit=QUDT_UNIT.PIXEL),
            'ny': StandardName(standardName='sensor_pixel_height', unit=QUDT_UNIT.PIXEL),
            'bit_depth': StandardName(standardName='sensor_bit_depth', unit=QUDT_UNIT.BIT),
            'fill_ratio_x': StandardName(standardName='sensor_pixel_width_fill_factor', unit=QUDT_UNIT.UNITLESS),
            'fill_ratio_y': StandardName(standardName='sensor_pixel_height_fill_factor', unit=QUDT_UNIT.UNITLESS),
            'particle_image_diameter': StandardName(standardName='image_particle_diameter', unit=QUDT_UNIT.M)
        }
        descr_dict = {
            'qe': 'quantum efficiency',
            'dark_noise': 'Dark noise is the standard deviation of a gaussian noise model',
            'baseline_noise': 'Dark noise is the mean value of a gaussian noise model'
        }
        unit_dict = {
            'nx': QUDT_UNIT.UNITLESS,
            'ny': QUDT_UNIT.UNITLESS,
            'bit_depth': QUDT_UNIT.BIT,  # 'http://qudt.org/vocab/unit/BIT',
        }
        qkind_dict = {
            'nx': QUDT_KIND.Dimensionless,
            'ny': QUDT_KIND.Dimensionless,
            'bit_depth': QUDT_KIND.InformationEntropy  # 'http://qudt.org/schema/qudt/CountingUnit'
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
            m4i.TextVariable(label='shot_noise',
                                      hasStringValue=shot_noise_txt_value)
        )
        camera = pivmeta.VirtualCamera(
            hasSourceCode=get_package_meta(),
            hasParameter=hasParameter
        )
        return camera.model_dump_jsonld(
            context={
                "@import": 'https://raw.githubusercontent.com/matthiasprobst/pivmeta/main/pivmeta_context.jsonld',
                # "codemeta": 'https://codemeta.github.io/terms/'
            }
        )

    def save_jsonld(self, filename: Union[str, pathlib.Path]):
        """Save the component to JSON"""
        filename = pathlib.Path(filename)  # .with_suffix('.jsonld')
        with open(filename, 'w') as f:
            f.write(
                self.model_dump_jsonld()
            )
        return filename

    @classmethod
    def load_jsonld(cls, filename: Union[str, pathlib.Path]):
        """Load the camera from a JSON-LD file

        .. note::

            This function will return a list of Camera objects. This may be
            confusing, but there might be multiple lasers in the JSON file.

        Parameters
        ----------
        filename : Union[str, pathlib.Path]
            The filename to load the component from

        Returns
        -------
        List[Camera]
            List of camera objects
        """
        return load_jsonld(cls, 'pivmeta:VirtualCamera', filename)
