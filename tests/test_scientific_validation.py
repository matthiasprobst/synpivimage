import unittest

import numpy as np

import synpivimage
from synpivimage.camera import Camera
from synpivimage.core import take_image
from synpivimage.laser import Laser
from synpivimage.noise import compute_dark_noise
from synpivimage.particles import Particles


class TestScientificValidation(unittest.TestCase):

    def _build_camera(self, **kwargs) -> Camera:
        params = dict(
            nx=32,
            ny=32,
            bit_depth=16,
            qe=1.0,
            sensitivity=1.0,
            baseline_noise=0.0,
            dark_noise=0.0,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=2.0,
        )
        params.update(kwargs)
        return Camera(**params)

    @staticmethod
    def _build_particle(z_value: float = 0.0, size: float = 2.0) -> Particles:
        return Particles(
            x=np.array([16.0]),
            y=np.array([16.0]),
            z=np.array([z_value]),
            size=np.array([size]),
        )

    def test_noise_sequence_is_reproducible_and_non_repeating(self):
        laser = Laser(width=5.0, shape_factor=1000)
        camera_a = self._build_camera(
            shot_noise=True,
            dark_noise=3.0,
            baseline_noise=10.0,
            seed=123,
        )

        img_a0, _ = take_image(laser, camera_a, self._build_particle(), particle_peak_count=600)
        img_a1, _ = take_image(laser, camera_a, self._build_particle(), particle_peak_count=600)

        # The RNG state advances between captures.
        self.assertFalse(np.array_equal(img_a0, img_a1))

        # A new camera with the same seed reproduces the first realization.
        camera_b = self._build_camera(
            shot_noise=True,
            dark_noise=3.0,
            baseline_noise=10.0,
            seed=123,
        )
        img_b0, _ = take_image(laser, camera_b, self._build_particle(), particle_peak_count=600)
        np.testing.assert_array_equal(img_a0, img_b0)

    def test_dark_noise_uses_local_rng_and_is_not_clipped(self):
        np.random.seed(1)
        noise_a = compute_dark_noise(
            mean=0.0,
            std=3.0,
            shape=(128, 128),
            rs=np.random.RandomState(11),
        )

        np.random.seed(999)
        noise_b = compute_dark_noise(
            mean=0.0,
            std=3.0,
            shape=(128, 128),
            rs=np.random.RandomState(11),
        )

        np.testing.assert_allclose(noise_a, noise_b)
        self.assertTrue(np.any(noise_a < 0))

    def test_out_of_plane_transition_is_continuous(self):
        camera = self._build_camera()
        laser = Laser(width=1.0, shape_factor=2)
        particles = self._build_particle(z_value=0.8, size=2.0)

        _, part = take_image(
            laser=laser,
            camera=camera,
            particles=particles,
            particle_peak_count=1000,
        )

        self.assertEqual(part.in_fov.sum(), 1)
        self.assertEqual(part.active.sum(), 1)

    def test_polydisperse_peak_is_calibrated_to_largest_particle_response(self):
        camera = self._build_camera()
        laser = Laser(width=10.0, shape_factor=1000)
        particles = Particles(
            x=np.array([8.0, 24.0]),
            y=np.array([8.0, 24.0]),
            z=np.array([0.0, 0.0]),
            size=np.array([1.0, 4.0]),
        )

        _, part = take_image(
            laser=laser,
            camera=camera,
            particles=particles,
            particle_peak_count=1000,
        )

        self.assertGreater(part.max_image_photons[1], part.max_image_photons[0])
        self.assertAlmostEqual(part.max_image_photons.max(), 1000, delta=3)

    def test_defocus_reduces_peak_intensity(self):
        laser = Laser(width=10.0, shape_factor=1000)
        particles = self._build_particle(z_value=2.0, size=2.0)

        camera_ref = self._build_camera(defocus_strength=0.0, focus_plane_z=0.0)
        camera_defocus = self._build_camera(defocus_strength=0.6, focus_plane_z=0.0)

        img_ref, _ = take_image(laser, camera_ref, particles.copy(), particle_peak_count=1000)
        img_defocus, _ = take_image(laser, camera_defocus, particles.copy(), particle_peak_count=1000)

        self.assertLess(img_defocus.max(), img_ref.max())


class TestMetadataUnits(unittest.TestCase):

    def test_laser_width_metadata_units_are_consistent(self):
        laser = synpivimage.Laser(width=1.0, shape_factor=2)
        jsonld = laser.model_dump_jsonld()

        self.assertIn("model_laser_sheet_thickness", jsonld)
        self.assertIn("http://qudt.org/vocab/unit/MilliM", jsonld)
