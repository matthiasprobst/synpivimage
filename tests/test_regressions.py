import pathlib
import unittest

import h5py
import numpy as np

import synpivimage
from synpivimage.camera import Camera
from synpivimage.laser import Laser
from synpivimage.particles import Particles

__this_dir__ = pathlib.Path(__file__).parent


class TestParticlesRegressions(unittest.TestCase):

    def test_particle_property_setters(self):
        particles = Particles(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([0.0, 0.0, 0.0]),
            size=np.array([2.0, 2.0, 2.0]),
        )

        max_photons = np.array([10.0, 20.0, 30.0])
        electrons = np.array([5.0, 6.0, 7.0])
        quantized = np.array([1.0, 2.0, 3.0])
        source = np.array([100.0, 200.0, 300.0])

        particles.max_image_photons = max_photons
        particles.image_electrons = electrons
        particles.image_quantized_electrons = quantized
        particles.irrad_photons = source

        np.testing.assert_array_equal(particles.max_image_photons, max_photons)
        np.testing.assert_array_equal(particles.image_electrons, electrons)
        np.testing.assert_array_equal(particles.image_quantized_electrons, quantized)
        np.testing.assert_array_equal(particles.source_intensity, source)

    def test_displace_requires_imaged_particles(self):
        particles = Particles(
            x=np.array([8.0]),
            y=np.array([8.0]),
            z=np.array([0.0]),
            size=np.array([2.0]),
        )

        with self.assertRaises(ValueError):
            particles.displace(dx=1.0)

    def test_generate_uniform_returns_particles(self):
        particles = Particles.generate_uniform(
            n_particles=100,
            size=2.0,
            x_bounds=(-1.0, 17.0),
            y_bounds=(-1.0, 17.0),
            z_bounds=(-0.5, 0.5),
        )

        self.assertIsInstance(particles, Particles)
        self.assertEqual(len(particles), 100)


class TestHdfWriterRegressions(unittest.TestCase):

    def setUp(self):
        self.filename = __this_dir__ / "regression_particles.hdf"
        self.filename.unlink(missing_ok=True)

    def tearDown(self):
        self.filename.unlink(missing_ok=True)

    def test_particles_written_at_requested_index(self):
        cam = Camera(
            nx=16,
            ny=16,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=0,
            dark_noise=0,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=2,
        )
        laser = Laser(width=1.0, shape_factor=2)

        particles = Particles(
            x=np.array([8.0, 9.0]),
            y=np.array([8.0, 8.0]),
            z=np.array([0.0, 0.0]),
            size=np.array([2.0, 2.0]),
        )
        img, part = synpivimage.take_image(
            laser=laser,
            camera=cam,
            particles=particles,
            particle_peak_count=1000,
        )

        with synpivimage.HDF5Writer(
            filename=self.filename,
            n_images=2,
            overwrite=True,
            camera=cam,
            laser=laser,
        ) as writer:
            writer.writeA(1, img, particles=part)

        with h5py.File(self.filename, "r") as h5:
            ds = h5["particles/A/x"]
            self.assertEqual(ds.shape, (2, len(part)))
            np.testing.assert_allclose(ds[1, :], part.x)
            np.testing.assert_allclose(ds[0, :], 0.0)
