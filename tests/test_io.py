import pathlib
import unittest

import h5py
import numpy as np
from pivmetalib.pivmeta import VirtualLaser

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
        gauss_laser_filename = gauss_laser.save_jsonld(__this_dir__ / 'laser.json')

        loaded_laser = VirtualLaser.from_jsonld(gauss_laser_filename)[0]

        self.assertIsInstance(loaded_laser, VirtualLaser)

        with open(__this_dir__ / 'laser2.json', 'w') as f:
            f.write(loaded_laser.model_dump_jsonld())

        self.assertEqual(len(loaded_laser.hasParameter),
                         2)
        self.assertEqual(loaded_laser.hasParameter[0].label,
                         'width')
        self.assertEqual(loaded_laser.hasParameter[0].hasStandardName.standardName,
                         'model_laser_sheet_thickness')
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

    def test_write_hdf(self):
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
        laser = synpivimage.Laser(
            width=1,
            shape_factor=1
        )
        particles = Particles(
            x=np.random.uniform(-5, cam.nx - 1, 40),
            y=np.random.uniform(-10, cam.ny - 1, 40),
            z=np.random.uniform(-1, 1, 40),
            size=np.ones(40) * 2,
        )
        img, _ = synpivimage.take_image(laser=laser,
                                        camera=cam,
                                        particles=particles,
                                        particle_peak_count=1000)

        with synpivimage.HDF5Writer(__this_dir__ / 'data.hdf',
                                    n_images=2,
                                    overwrite=True,
                                    camera=cam,
                                    laser=laser) as h5:
            h5.writeA(0, img=img)
            h5.writeA(1, img=img)
            with self.assertRaises(KeyError):
                h5.writeA(2, img=img)

        with h5py.File(__this_dir__ / 'data.hdf', 'r') as h5:
            self.assertEqual(h5['images/img_A'].shape, (2, 16, 16))
            self.assertEqual(h5['images/img_A'].dtype, np.uint16)
            self.assertEqual(h5['images/image_index'].shape, (2,))
            self.assertEqual(h5['images/ix'].shape, (cam.nx,))
            self.assertEqual(h5['images/iy'].shape, (cam.ny,))
            self.assertEqual(h5['images/image_index'][0], 0)
            self.assertEqual(h5['images/image_index'][1], 1)
            np.testing.assert_array_equal(h5['images/ix'][()], np.arange(cam.nx))
            np.testing.assert_array_equal(h5['images/iy'][()], np.arange(cam.ny))
