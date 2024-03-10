import logging
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from ontolutils.namespacelib import PIVMETA, QUDT_UNIT
from pydantic import BaseModel

from .codemeta import get_package_meta
from .component import Component, load_jsonld
from .particles import Particles
from .validation import PositiveInt, PositiveFloat

LOGGER = logging.getLogger('synpivimage')

SQRT2 = np.sqrt(2)
SQRT2pi = np.sqrt(2 * np.pi)
DEBUG_LEVEL = 0


class real:

    def __init__(self, dz0, s):
        self.dz0 = dz0
        self.s = s

    def __call__(self, z):
        return np.exp(-1 / SQRT2pi * np.abs(2 * z ** 2 / self.dz0 ** 2) ** self.s)


class tophat:
    """Tophat function"""

    def __init__(self, dz0):
        self.dz0 = dz0

    def __call__(self, z) -> np.ndarray:
        intensity = np.ones_like(z)
        intensity[z < -self.dz0 / 2] = 0
        intensity[z > self.dz0 / 2] = 0
        return intensity


def const(z):
    """Const laser. No sheet, illuminates all the particles."""
    return np.ones_like(z)


class Laser(BaseModel, Component):
    """Laser class. This class will be used to illuminate the particles.

    .. note::

        The Gaussian distribution is found for shape_factor=1, not =2 as
        in the literature (which is wrong, e.g. see Raffel et al.)!
        width is the width of the laser, where the intensity drops to 0.67, i.e.
        not the effective laser width, with is defined by the noise level or where
        the intensity drops to e^(-1).
    """
    shape_factor: PositiveInt
    width: PositiveFloat  # width of the laser, not the effective laser width

    def illuminate(self,
                   particles: Particles,
                   **kwargs) -> Particles:
        """Illuminate the particles. The values will be between 0 and 1.
        Particles outside the laser will be masked.

        Parameters
        ----------
        particles : Particles
            The particles to be illuminated
        kwargs : dict
            Additional parameters

        Returns
        -------
        Particles
            The illuminated particles (new object!)
        """
        logger = kwargs.get('logger', LOGGER)

        # the width of a laser is defined as:
        # intensity drops to 1-e

        dz0 = SQRT2 * self.width / 2
        s = self.shape_factor
        if s == 0:
            laser_intensity = const
        elif s > 100:
            laser_intensity = tophat(self.width)
        else:
            laser_intensity = real(dz0, s)
        particles.source_intensity = laser_intensity(particles.z)

        inside_laser = particles.source_intensity > np.exp(-2)

        particles.mask = inside_laser  # mask for the particles inside the laser

        if DEBUG_LEVEL > 0:
            n_removed = np.sum(~inside_laser)
            n_total = len(particles)
            perc_removed = n_removed / n_total * 100
            logger.debug(f'Removed {n_removed} ({perc_removed} %) particles because they are outside the laser,'
                         f' which is defined as an intensity below exp(-2)')

        if DEBUG_LEVEL > 1:
            plt.figure()
            plt.plot(particles.z[inside_laser], particles.source_intensity[inside_laser], 'o', color='g')
            plt.plot(particles.z[~inside_laser], particles.source_intensity[~inside_laser], 'o', color='r')
            plt.xlabel('z / real arbitrary units')
            plt.ylabel('Normalized particle intensity in beam / -')
            plt.grid()
            plt.show()
        return Particles(**particles.dict())

    def save_jsonld(self, filename: Union[str, pathlib.Path]) -> pathlib.Path:
        """Save the component to JSON.

        Parameters
        ----------
        filename : Union[str, pathlib.Path]
            The filename to save the component to

        Returns
        -------
        pathlib.Path
            The filename the component was saved to
        """
        try:
            from pivmetalib import pivmeta
        except ImportError:
            raise ImportError("Please install `pivmetalib` to use this function: `pip install pivmetalib`")

        filename = pathlib.Path(filename)  # .with_suffix('.jsonld')
        laser = pivmeta.LaserModel(
            hasParameter=[
                pivmeta.NumericalVariable(
                    label='width',
                    hasNumericalValue=self.width,
                    hasStandardName=PIVMETA.model_laser_sheet_thickness,
                    hasUnit='mm',
                    hasKindOfQuantity=QUDT_UNIT.MilliM,  # 'https://qudt.org/vocab/unit/MilliM',
                    hasVariableDescription='Laser width'),
                pivmeta.NumericalVariable(
                    label='shape_factor',
                    hasNumericalValue=self.shape_factor,
                    hasStandardName=PIVMETA.model_laser_sheet_shape_factor,
                    hasUnit='',
                    hasKindOfQuantity="https://qudt.org/schema/qudt/DimensionlessUnit",
                    hasVariableDescription='The shape factor describes they laser beam shape. A '
                                           'value of 1 describes Gaussian beam shape. '
                                           'High value are top-hat-like shapes.'),
            ],
            hasSourceCode=get_package_meta(),
        )
        with open(filename, 'w') as f:
            f.write(
                laser.model_dump_jsonld(context={'local': 'http://example.org/'})
            )
        return filename

    @classmethod
    def load_jsonld(cls, filename: Union[str, pathlib.Path]):
        """Load the Laser from a JSON-LD file

        .. note::

            This function will return a list of Laser objects. This may be
            confusing, but there might be multiple lasers in the JSON file.

        Parameters
        ----------
        filename : Union[str, pathlib.Path]
            The filename to load the component from

        Returns
        -------
        List[Laser]
            List of laser objects
        """
        return load_jsonld(cls, 'pivmeta:LaserModel', filename)


class GaussShapeLaser(Laser):
    """Gaussian laser"""

    def __init__(self, width: float):
        super().__init__(shape_factor=1, width=width)
