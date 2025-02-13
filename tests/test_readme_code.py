import pathlib
import shutil
import unittest

import numpy as np

import synpivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestReadmeCode(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree('test_case', ignore_errors=True)

    def test_code(self):
        cam = synpivimage.Camera(
            nx=256,
            ny=256,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=50,
            dark_noise=10,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=4  # px
        )

        laser = synpivimage.Laser(
            width=0.25,
            shape_factor=2
        )

        n = 100
        particles = synpivimage.Particles(
            x=np.random.uniform(-3, cam.nx - 1, n),
            y=np.random.uniform(-4, cam.ny - 1, n),
            z=np.zeros(n),
            size=np.ones(n) * 2,
        )

        imgA, partA = synpivimage.take_image(laser,
                                             cam,
                                             particles,
                                             particle_peak_count=1000)

        displaced_particles = partA.displace(dx=2.1, dy=3.4)

        imgB, partB = synpivimage.take_image(laser,
                                             cam,
                                             displaced_particles,
                                             particle_peak_count=1000)

        with synpivimage.Imwriter(case_name="test_case",
                                  camera=cam,
                                  laser=laser,
                                  overwrite=True) as iw:
            iw.writeA(0, imgA, partA)
            iw.writeB(0, imgB, partB)

        with synpivimage.HDF5Writer(filename=__this_dir__ / 'data.hdf',
                                    n_images=1,
                                    camera=cam,
                                    laser=laser,
                                    overwrite=True) as hw:
            hw.writeA(0, imgA, partA)
            hw.writeB(0, imgB, partB)
