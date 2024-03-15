import logging
import pathlib
import shutil
import unittest

import numpy as np

from synpivimage import io
from synpivimage import take_image
from synpivimage.camera import Camera
from synpivimage.laser import Laser
from synpivimage.particles import Particles, ParticleFlag

__this_dir__ = pathlib.Path(__file__).parent
logger = logging.getLogger('synpivimage')
logger.setLevel('DEBUG')
for h in logger.handlers:
    h.setLevel('DEBUG')


class TestLaser(unittest.TestCase):

    def test_save_load_camera(self):
        laser = Laser(
            width=0.25,
            shape_factor=2
        )
        filename = laser.save_jsonld(__this_dir__ / 'laser.json')
        laser_loaded = Laser.load_jsonld(__this_dir__ / 'laser.json')[0]
        self.assertEqual(laser, laser_loaded)
        filename.unlink(missing_ok=True)

    def test_effective_laser_width(self):
        """Test the effective laser width"""
        laser = Laser(
            width=0.25,
            shape_factor=2
        )
        cam = Camera(
            nx=16,
            ny=16,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=200,
            dark_noise=100,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=1.0
        )
        n = 400
        many_particles = Particles(
            x=np.ones(n) * cam.nx // 2,
            y=np.ones(n) * cam.ny // 2,
            z=np.linspace(-3 * laser.width / 2, 3 * laser.width / 2, n),
            size=np.ones(n) * 2
        )
        imgOne, partOne = take_image(laser, cam, many_particles, particle_peak_count=1000)

        ppp = partOne.get_ppp(cam.size)
        self.assertIsInstance(ppp, float)
        ppp_ref = partOne.active.sum() / cam.size
        self.assertAlmostEqual(ppp, ppp_ref, delta=0.01)


class TestParticles(unittest.TestCase):

    def test_regenerate(self):
        particles = Particles(
            x=4, y=4, z=0, size=2
        )
        particles.regenerate()
        self.assertEqual(particles.x, 4)
        self.assertEqual(particles.y, 4)

        particles = Particles(
            x=[4, 10], y=[4, 10], z=[0, 0], size=[2, 2]
        )
        self.assertEqual(particles.x[0], 4)
        self.assertEqual(particles.x[1], 10)
        new_part = particles.regenerate()
        self.assertNotEqual(particles.x[0], 4)
        self.assertNotEqual(particles.x[1], 10)

        self.assertTrue(new_part, particles)

    def test_compute_ppp(self):
        laser = Laser(
            width=0.25,
            shape_factor=2
        )
        cam = Camera(
            nx=16,
            ny=16,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=200,
            dark_noise=100,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=1.0
        )
        particles = Particles(
            x=4, y=4, z=0, size=2
        )
        imgOne, partOne = take_image(laser, cam, particles, particle_peak_count=1000)
        self.assertTrue(particles is partOne)

        self.assertEqual(partOne.get_ppp(camera_size=cam.size), 1 / cam.size)

        particles = Particles(
            x=[2, 4, 8], y=[2, 4, 8], z=[0, 0, 0], size=[2, 2, 2]
        )
        self.assertIsInstance(particles.x, np.ndarray)
        self.assertEqual(particles.x.size, 3)
        self.assertEqual(particles.x.shape, (3,))
        np.testing.assert_array_equal(particles.x, np.array([2, 4, 8]))

        imgOne, partOne = take_image(laser, cam, particles, particle_peak_count=1000)
        self.assertEqual(partOne.get_ppp(camera_size=cam.size), 3 / cam.size)

        particles = Particles(
            x=[-10, 4, 8], y=[2, 4, 8], z=[0, 0, 0], size=[2, 2, 2]
        )

        imgOne, partOne = take_image(laser, cam, particles, particle_peak_count=1000)
        self.assertEqual(partOne.get_ppp(camera_size=cam.size), 2 / cam.size)

        particles = Particles(
            x=[2, 4, 8], y=[2, 4, 8], z=[0, 10, 0], size=[2, 2, 2]
        )

        imgOne, partOne = take_image(laser, cam, particles, particle_peak_count=1000)
        self.assertEqual(partOne.get_ppp(camera_size=cam.size), 2 / cam.size)

    def test_generate_certain_ppp(self):
        laser = Laser(shape_factor=10 ** 3, width=10)

        cam = Camera(
            nx=128,
            ny=128,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=0,
            dark_noise=0,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=2,
            seed=10
        )
        target_ppp = 0.01
        partA = Particles.generate(ppp=target_ppp,
                                   dx_max=[10, 100],
                                   dy_max=[-100, 100],
                                   dz_max=[1, 1],
                                   size=2,
                                   camera=cam,
                                   laser=laser)
        print(partA.get_ppp(cam.size))
        # check if crit is really reached (err < 0.01):
        realized_ppp = partA.get_ppp(cam.size)
        err = abs((realized_ppp - target_ppp) / target_ppp)
        self.assertTrue(err <= 0.01)

    def test_generate_certain_ppp_with_noise(self):
        laser = Laser(shape_factor=10 ** 3, width=10)

        cam = Camera(
            nx=128,
            ny=128,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=0,
            dark_noise=0,
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=4,
            seed=10
        )

        true_dx = 0.6
        true_dy = 0.3
        ppp_list = np.linspace(0.001, 0.1, 11, dtype='float32')
        dark_noise_list = np.linspace(0, 100, 6, dtype='float32')
        for ppp in ppp_list[:]:
            for dark_noise in dark_noise_list:
                cam.dark_noise = dark_noise
                particles = Particles.generate(
                    ppp=ppp,
                    dx_max=[0, true_dx],
                    dy_max=[0, true_dy],
                    dz_max=[0, 0],
                    size=2,
                    camera=cam,
                    laser=laser)

                realized_ppp = particles.get_ppp(cam.size)
                err = abs((realized_ppp - ppp) / ppp)
                self.assertTrue(err <= 0.05, msg=f'ppp: {ppp}, realized_ppp: {realized_ppp}, err: {err}')

    def test_too_much_noise(self):
        laser = Laser(shape_factor=10 ** 3, width=10)

        cam = Camera(
            nx=128,
            ny=128,
            bit_depth=16,
            qe=1,
            sensitivity=1,
            baseline_noise=0,
            dark_noise=1000,  # too much dark noise!
            shot_noise=False,
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=2,
            seed=10
        )
        target_ppp = 0.01
        with self.assertRaises(ValueError):
            partA = Particles.generate(ppp=target_ppp,
                                       dx_max=[10, 100],
                                       dy_max=[-100, 100],
                                       dz_max=[1, 1],
                                       size=2,
                                       camera=cam,
                                       laser=laser)

    def test_single_particle(self):
        one_particle = Particles(
            x=1,
            y=1,
            z=1,
            size=1,
        )
        self.assertEqual(len(one_particle), 1)
        self.assertEqual(one_particle.x, np.array([1, ]))

        one_particle = Particles(
            x=[1, ],
            y=[1, ],
            z=[1, ],
            size=[1, ]
        )
        self.assertEqual(len(one_particle), 1)
        self.assertEqual(one_particle.x, np.array([1, ]))
        with self.assertRaises(AttributeError):
            one_particle.x = 3.

        with self.assertRaises(AttributeError):
            one_particle[0].x = 3.

        self.assertEqual(one_particle[0].x, np.array([1, ]))

        with self.assertRaises(ValueError):
            one_particle = Particles(
                x=np.array([1, ]),
                y=np.array([1, 2, 3]),
                z=np.array([1, ]),
                size=np.array([1, ])
            )

    def test_init_particles(self):
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
            particle_image_diameter=2
        )
        n = 10
        many_particles = Particles(
            x=np.ones(n) * cam.nx // 2,
            y=np.ones(n) * cam.ny // 2,
            z=np.linspace(-3, 3, n),
            size=np.ones(n) * 2
        )


class TestCore(unittest.TestCase):

    def setUp(self) -> None:
        self.cam = Camera(
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
            particle_image_diameter=2)
        self.laser = Laser(
            width=0.25,
            shape_factor=1000
        )

    def test_source_density(self):
        """
        Source density:
        Ns = C*...

        """

    def test_max_particle_count(self):
        """If the user specifies particle_peak_count=1000, then a single particle in
        the center of the light sheet should have the max count of 1000"""

        assert self.cam.dark_noise == 0
        assert not self.cam.shot_noise

        imgA, partA = take_image(self.laser, self.cam,
                                 Particles(x=8, y=8, z=0, size=2),
                                 1000)
        self.assertAlmostEqual(imgA.max(), 1000, delta=1)
        self.assertAlmostEqual(partA.max_image_photons.max(), 1000, delta=1)

    def test_out_of_plane(self):
        imgA, partA = take_image(self.laser, self.cam,
                                 Particles(x=8, y=8, z=0, size=2),
                                 1000)
        self.assertEqual(partA.active.sum(), 1)
        self.assertEqual(partA.n_out_of_plane_loss, 0)
        self.assertEqual(partA.in_fov.sum(), 1)

        imgB, partB = take_image(self.laser, self.cam,
                                 partA.displace(dz=10), 1000)
        self.assertEqual(partA.active.sum(), 1)
        self.assertEqual(partB.n_out_of_plane_loss, 1)
        self.assertEqual(partB.in_fov.sum(), 1)

    def test_take_image(self):
        # place particles outside laser sheet
        particles = Particles(
            x=8,
            y=8,
            z=-100,
            size=2
        )
        imgA, particlesA = take_image(self.laser, self.cam, particles, 1000)
        self.assertEqual(0, np.asarray(particlesA.flag & ParticleFlag.ILLUMINATED.value, dtype=bool).sum())

        # distinct position (middle)
        particles = Particles(
            x=8,
            y=8,
            z=0,
            size=2
        )

        particle_peak_count = 1000

        imgA, particlesA = take_image(self.laser, self.cam, particles, particle_peak_count)
        self.assertAlmostEqual(imgA.max(), particle_peak_count, delta=1)
        self.assertEqual(1, np.asarray(particlesA.flag & ParticleFlag.ILLUMINATED.value, dtype=bool).sum())

        # save multiples:
        iw = io.Imwriter(image_dir=__this_dir__, case_name='test_imgs_01', camera=self.cam, laser=self.laser,
                         overwrite=True)
        with iw as imwriter:
            for i in range(10):
                imgA, particlesA = take_image(self.laser, self.cam, particles, particle_peak_count)
                imwriter.writeA(i, imgA, particlesA)
                imwriter.writeB(i, imgA, particlesA)

        self.assertTrue(iw.img_filenames[0].exists())
        self.assertTrue(iw.img_filenames[0].is_file())
        self.assertEqual(iw.img_filenames[0].parent.name, 'imgs')
        self.assertEqual(iw.img_filenames[0].suffix, '.tif')

        shutil.rmtree(iw.image_dir)

        pathlib.Path(pathlib.Path('img01a.tiff')).unlink(missing_ok=True)
        pathlib.Path(pathlib.Path('img01a.hdf')).unlink(missing_ok=True)
        pathlib.Path(pathlib.Path('img01a.json')).unlink(missing_ok=True)

#
#
# class TestCore(unittest.TestCase):
#
#     def setUp(self) -> None:
#         """setup"""
#         hdf_filenames = __this_dir__.glob('ds*.hdf')
#         for hdf_filename in hdf_filenames:
#             hdf_filename.unlink(missing_ok=True)
#
#     def test_particle_size_definition(self):
#         """Take a single mage at (x0, y0) = (9, 9)"""
#         cfg_single_particle = SynPivConfig(
#             ny=20,
#             nx=20,
#             bit_depth=8,
#             dark_noise=0,
#             image_particle_peak_count=10,
#             laser_shape_factor=2,
#             laser_width=2,
#             noise_baseline=0,
#             particle_number=10,
#             particle_size_illumination_dependency=True,
#             particle_size_mean=10,  # WILL BE OVERWRITTEN by particle_data
#             particle_size_std=0,
#             sigmax=2.0,
#             sigmay=2.0,
#             fill_ratio_x=1.0,
#             fill_ratio_y=1.0,
#             qe=1.,
#             sensitivity=1.,
#             shot_noise=False)
#
#         img, _, _ = generate_image(
#             cfg_single_particle,
#             particle_data=ParticleInfo(x=9, y=9, z=0, size=4)
#         )
#
#         plt.figure()
#         plt.imshow(img)
#         plt.show()
#
#         pixel_values = img[:, 9]
#         ix = np.arange(0, 20, 1)
#         x0 = 9
#
#         plt.figure()
#         plt.scatter(ix - x0, pixel_values, label='pixel values')
#
#         def gauss(x, I0, pattern_meanx):
#             """Simple 1D form. psize=2*sigma. We know that x0=9"""
#             x0 = 9
#             return I0 * np.exp(-((x - x0) ** 2) / (2 * pattern_meanx ** 2))
#
#         n_pts = 5
#         popt, pcov = curve_fit(gauss, ix[x0 - n_pts:x0 + n_pts + 1], pixel_values[x0 - n_pts:x0 + n_pts + 1])
#         plt.scatter(ix[x0 - n_pts:x0 + n_pts + 1] - x0, pixel_values[x0 - n_pts:x0 + n_pts + 1],
#                     label='fitting pts')
#         ix_interp = np.linspace(x0 - n_pts, x0 + n_pts, 100)
#         plt.plot(ix_interp - x0, gauss(ix_interp, *popt), label='fit')
#         _, pattern_meanx = popt
#         print('guess for pattern_meanx:', pattern_meanx)
#         ymax = plt.gca().get_ylim()[1]
#         plt.vlines(-pattern_meanx, 0, ymax)
#         plt.vlines(pattern_meanx, 0, ymax)
#         _ = plt.legend()
#         plt.show()
#
#         # per definition: particle size = 2 * sigma!
#         self.assertAlmostEqual(round(abs(pattern_meanx), 1), round(cfg_single_particle.pattern_meanx, 1), 0)
#
#     def test_write_read_config(self):
#         from synpivimage.core import generate_default_yaml_file, read_config, SynPivConfig
#         filename = generate_default_yaml_file()
#
#         cfg = read_config(filename)
#         self.assertIsInstance(cfg, SynPivConfig)
#
#         filename.unlink(missing_ok=True)
#
#     def test_build_config_manager(self):
#         cfg = get_default()
#         cfg.nx = 16
#         cfg.ny = 16
#         particle_number_variation = dict(particle_number=np.linspace(1, cfg.ny * cfg.nx, 101).astype(int))
#         CFG = build_ConfigManager(initial_cfg=cfg,
#                                   variation_dict=particle_number_variation,
#                                   per_combination=1)
#         assert len(CFG) == 101
#         CFG = build_ConfigManager(initial_cfg=cfg,
#                                   variation_dict=particle_number_variation,
#                                   per_combination=2)
#         assert len(CFG) == 101 * 2
#
#         generated_particle_number = [cfg['particle_number'] for cfg in CFG.cfgs]
#         assert np.array_equal(np.unique(np.sort(generated_particle_number)),
#                               particle_number_variation['particle_number'])
#
#     def test_generate(self):
#         cfg = get_default()
#         cfg.nx = 16
#         cfg.ny = 16
#         particle_number_range = {'particle_number': np.linspace(1, cfg.ny * cfg.nx, 5).astype(int)}
#         CFG = build_ConfigManager(initial_cfg=cfg,
#                                   variation_dict=particle_number_range,
#                                   per_combination=1)
#         CFG.generate('.', nproc=1)
#
#         hdf_filename = 'ds_000000.hdf'
#
#         with h5py.File(hdf_filename) as h5:
#             self.assertIn('images', h5)
#             self.assertIn('labels', h5)
#             self.assertEqual(h5['labels'].shape, h5['images'].shape)
#             np.testing.assert_almost_equal(h5['nparticles'][:], (h5['labels'][...].sum(axis=(1, 2)) / 100).astype(int))
#             self.assertIn('image_index', h5)
#             self.assertTrue(h5['images'].attrs['standard_name'], 'synthetic_particle_image')
#             for ds_name in h5.keys():
#                 if isinstance(h5[ds_name], h5py.Dataset) and ds_name == 'images':
#                     assert h5[ds_name].dims[0][0] == h5['image_index']
#                     assert h5[ds_name].dims[0][1] == h5['nparticles']
#             assert h5['images'].dims[1][0] == h5['iy']
#             assert h5['images'].dims[2][0] == h5['ix']
#
#     def test_generate_second_image(self):
#         cfg = get_default()
#         cfg.nx = 16
#         cfg.ny = 16
#         cfg.laser_width = 2
#         particle_numbers = np.linspace(1, cfg.ny * cfg.nx * 0.1, 5).astype(int)
#
#         CFG = build_ConfigManager(
#             initial_cfg=cfg,
#             variation_dict={'particle_number': particle_numbers},
#             per_combination=1
#         )
#
#         # with h5tbx.File(hdf_filenames[0]) as h5:
#         #     get x,y,z,size from hdf file and feed to to image B generation
#
#         hdf_filenamesA = CFG.generate(
#             data_directory='.',
#             nproc=1,
#             suffix='A.hdf',
#             particle_info=synpivimage.core.ParticleInfo(
#                 x=np.array([8, 9, 10, 11, 12]),
#                 y=np.array([8, 8, 8, 8, 8]),
#                 z=np.array([0, -0.5, -1, -1.5, -2]),
#                 size=np.array([2.5, 2.5, 2.5, 2.5, 2.5])
#             )
#         )
#         part_info = synpivimage.core.ParticleInfo.from_hdf(hdf_filenamesA[0])
#         [p.displace(dy=2, dx=0, dz=0) for p in part_info]
#         hdf_filenamesB = CFG.generate(
#             data_directory='.',
#             suffix='B.hdf',
#             nproc=1,
#             particle_info=part_info)
#
#         with h5tbx.File(hdf_filenamesA[0]) as h5:
#             h5.images[0, ...].plot()
#         plt.show()
#         with h5tbx.File(hdf_filenamesB[0]) as h5:
#             h5.images[0, ...].plot()
#         plt.show()
#
#     def test_create_single_image(self):
#         cfg = get_default()
#         cfg.nx = 16
#         cfg.ny = 16
#         cfg.laser_width = 2
#         cfg.particle_number = 1
#         cfg.qe = 1
#         cfg.dark_noise = 0
#         cfg.noise_baseline = 0
#         cfg.shot_noise = False
#         cfg.sensitivity = 1
#         cfg.bit_depth = 16
#         cfg.image_particle_peak_count = 1000
#
#         imgA, attrsA, part_infoA = generate_image(
#             cfg,
#             particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
#         )
#         self.assertEqual(imgA.max(), cfg.image_particle_peak_count)
#
#         cfg.qe = 0.25
#
#         imgA, attrsA, part_infoA = generate_image(
#             cfg,
#             particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
#         )
#         self.assertEqual(imgA.max(), cfg.image_particle_peak_count)
#
#         cfg.qe = 1
#         cfg.sensitivity = 1 / 4
#
#         imgA, attrsA, part_infoA = generate_image(
#             cfg,
#             particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
#         )
#         self.assertEqual(imgA.max(), cfg.image_particle_peak_count)
#
#         # max count = 1000
#         cfg.qe = 1
#         cfg.sensitivity = 1
#         cfg.image_particle_peak_count = 1000
#
#         imgA, attrsA, part_infoA = generate_image(
#             cfg,
#             particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
#         )
#         self.assertEqual(imgA.max(), cfg.image_particle_peak_count)
#         self.assertEqual(imgA[8, 8], cfg.image_particle_peak_count)
#
#         fig, axs = plt.subplots(1, 1)
#         imgAmax = imgA.max()
#         im = axs.imshow(imgA, cmap='gray', vmin=0, vmax=imgAmax)
#         plt.colorbar(im)
#         plt.show()
#
#     def test_out_of_plane(self):
#         cfg = get_default()
#         cfg.nx = 100
#         cfg.ny = 100
#         cfg.laser_width = 1
#         cfg.particle_number = 0.1 * cfg.nx * cfg.ny
#         imgA, attrsA, part_infoA = generate_image(
#             cfg
#         )
#         cfg.laser_shape_factor = 10 ** 3
#         print(cfg.particle_number)
#
#         imgB, attrsB, part_infoB = generate_image(
#             cfg,
#             particle_data=part_infoA.displace(dx=2, dy=1, dz=0.1)
#         )
#         print(attrsB)
#
#         def plot_img(img, ax):
#             im = ax.imshow(img, cmap='gray')
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.05)
#             cb = plt.colorbar(im, cax=cax)
#
#         fig, axs = plt.subplots(1, 2)
#         plot_img(imgA, axs[0])
#         plot_img(imgB, axs[1])
#         plt.show()
#
#     def test_displace_particles(self):
#         cfg = get_default()
#         cfg.nx = 16
#         cfg.ny = 16
#         cfg.laser_width = 1
#         cfg.particle_number = 5
#         cfg.qe = 0.25
#         cfg.dark_noise = 0
#         cfg.noise_baseline = 100
#         cfg.image_particle_peak_count = 1000
#         imgA, attrsA, part_infoA = generate_image(
#             cfg,
#             particle_data=synpivimage.ParticleInfo(
#                 x=8,
#                 y=8,
#                 z=0,
#                 size=2.5
#             )
#         )
#         new_part = part_infoA.displace(dx=2, dy=1, dz=-1)
#         imgB, attrsB, part_infoB = generate_image(
#             cfg,
#             particle_data=new_part
#         )
#
#         np.testing.assert_equal(part_infoB.x, part_infoA.x + 2)
#
#         fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
#         imgAmax = imgA.max()
#         im = axs[0].imshow(imgA, cmap='gray', vmin=0, vmax=imgAmax)
#
#         divider = make_axes_locatable(axs[0])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cb = plt.colorbar(im, cax=cax)
#
#         im = axs[1].imshow(imgB, cmap='gray', vmin=0, vmax=imgAmax)
#
#         divider = make_axes_locatable(axs[1])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cb = plt.colorbar(im, cax=cax)
#
#         plt.show()
#
#     def test_constant_displacement(self):
#         cfg = get_default()
#
#         cfg.bit_depth = 16
#         cfg.nx = 512
#         cfg.ny = 512
#         cfg.square_image = True
#
#         cfg.particle_size_mean = 2.5
#         cfg.particle_size_std = 0
#
#         cfg.particle_number = 1 / 64 * cfg.nx * cfg.ny
#         self.assertEqual(cfg.particle_number, 1 / 64 * 512 * 512)
#
#         cfg.image_particle_peak_count = 1000
#
#         cfg.dark_noise = 4  # std
#         cfg.noise_baseline = 100  # mean
#         cfg.shot_noise = True
#
#         cfg.qe = 1  # 1e-/count thus 4 baseline noise instead of 16
#         cfg.sensitivity = 1  # ADU/e-
#         cfg.laser_shape_factor = 10000
#         imgA, attrsA, part_infoA = generate_image(
#             cfg
#         )
#         self.assertEqual(len(part_infoA), cfg.particle_number)
#
#         cfield = velocityfield.ConstantField(dx=2.3, dy=1.6, dz=0)
#         displaced_particle_data = cfield.displace(cfg=cfg, part_info=part_infoA)
#
#         imgB, attrsB, part_infoB = generate_image(
#             cfg,
#             particle_data=displaced_particle_data
#         )
#
#     def test_displace_with_velocity_field(self):
#         cfg = get_default()
#
#         cfg.bit_depth = 16
#         cfg.nx = 512
#         cfg.ny = 512
#         cfg.square_image = True
#
#         cfg.particle_size_mean = 2.5
#         cfg.particle_size_std = 0
#
#         cfg.particle_number = 1 / 64 * cfg.nx * cfg.ny
#         self.assertEqual(cfg.particle_number, 1 / 64 * 512 * 512)
#
#         cfg.image_particle_peak_count = 1000
#
#         cfg.dark_noise = 4  # std
#         cfg.noise_baseline = 100  # mean
#         cfg.shot_noise = True
#
#         cfg.qe = 1  # 1e-/count thus 4 baseline noise instead of 16
#         cfg.sensitivity = 1  # ADU/e-
#         cfg.laser_shape_factor = 10000
#         imgA, attrsA, part_infoA = generate_image(
#             cfg
#         )
#         self.assertEqual(len(part_infoA), cfg.particle_number)
#
#         x = np.arange(-1, cfg.nx + 1, 1)
#         y = np.arange(-1, cfg.ny + 1, 1)
#         z = np.linspace(-cfg.laser_width - 1, cfg.laser_width + 1, 4)
#
#         randomfield = velocityfield.VelocityField(x=x,
#                                                   y=y,
#                                                   z=z,
#                                                   u=np.random.uniform(-1, 1, (len(z), len(y), len(x))),
#                                                   v=np.random.uniform(-1, 1, (len(z), len(y), len(x))),
#                                                   w=np.zeros((len(z), len(y), len(x)))
#                                                   )
#         new_loc = randomfield.displace(cfg, part_info=part_infoA)
