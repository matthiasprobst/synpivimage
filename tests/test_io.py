import pathlib
import unittest

import numpy as np
from pivmetalib.pivmeta import LaserModel

import synpivimage
from synpivimage.camera import Camera
from synpivimage.particles import Particles

__this_dir__ = pathlib.Path(__file__).parent


class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        self.filenames = []

    def tearDown(self) -> None:
        """delete created files"""
        for filename in self.filenames:
            pathlib.Path(filename).unlink(missing_ok=True)
        for filename in __this_dir__.glob('*.tiff'):
            filename.unlink(missing_ok=True)
        for filename in __this_dir__.glob('*.json'):
            filename.unlink(missing_ok=True)

    def test_io_laser(self):
        gauss_laser = synpivimage.Laser(
            width=1,
            shape_factor=1
        )
        gauss_laser_filename = gauss_laser.save_jsonld('laser.json')

        loaded_laser = LaserModel.from_jsonld(gauss_laser_filename)[0]

        self.assertIsInstance(loaded_laser, LaserModel)

        with open(__this_dir__ / 'laser2.json', 'w') as f:
            f.write(loaded_laser.model_dump_jsonld())

        self.assertEqual(len(loaded_laser.hasParameter),
                         2)
        self.assertEqual(loaded_laser.hasParameter[0].label,
                         'width')
        self.assertEqual(str(loaded_laser.hasParameter[0].hasStandardName),
                         'https://matthiasprobst.github.io/pivmeta#model_laser_sheet_thickness')
        self.assertEqual(loaded_laser.hasParameter[0].hasNumericalValue,
                         gauss_laser.width)

        (__this_dir__ / 'laser.json').unlink(missing_ok=True)
        (__this_dir__ / 'laser2.json').unlink(missing_ok=True)

    def test_write_particle_data(self):
        cam = Camera(
            nx=16,
            ny=16,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=50,
            dark_noise=10,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            sigmax=1,
            sigmay=1,
            particle_image_diameter=2,
        )
        cam.save_jsonld(filename='camera.json')
        pathlib.Path('camera.json').unlink(missing_ok=True)

        n = 40
        particles = Particles(
            x=np.random.uniform(-5, cam.nx - 1, n),
            y=np.random.uniform(-10, cam.ny - 1, n),
            z=np.random.uniform(-1, 1, n),
            size=np.ones(n) * 2,
        )
        particles.save_jsonld(filename='particles.json')

        pathlib.Path('particles.json').unlink(missing_ok=True)

        # self.filenames.append('particles.json')
        # metawrite(filename='particles.json',
        #           metadata=dict(particles=particles),
        #           format='json')
