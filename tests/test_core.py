import matplotlib.pyplot as plt
import unittest

from synpivimage import take_image
from synpivimage.camera import Camera
from synpivimage.laser import Laser
from synpivimage.particles import Particles


class TestCore(unittest.TestCase):

    def test_take_image(self):
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
            sigmax=1,
            sigmay=1,
        )

        laser = Laser(
            width=0.25,
            shape_factor=1002
        )

        # distinct position (middle)
        particles = Particles(
            x=8,
            y=8,
            z=0,
            size=2
        )

        particle_peak_count = 1000

        imgA = take_image(laser, cam, particles, particle_peak_count,
                          debug_level=2)
        self.assertEqual(imgA.max(), particle_peak_count)

        from synpivimage import io
        io.imwrite('img01a.tiff',
                   imgA[:],
                   cam=cam,
                   laser=laser,
                   overwrite=True)
        import h5py
        with h5py.File('img01a.hdf', 'w') as h5:
            ds = h5.create_dataset(
                'imgA',
                shape=(1, *imgA.shape),
                dtype=imgA.dtype)
            meta_group = h5.create_group('meta')
            laser_grp = meta_group.create_group('Laser')
            for k, v in laser.model_dump().items():
                laser_grp.create_dataset(k, shape=(1,))
            cam_grp = meta_group.create_group('Camera')
            for k, v in cam.model_dump().items():
                cam_grp.create_dataset(k, shape=(1,))
            io.hdfwrite(dataset=ds,
                        index=0,
                        img=imgA,
                        camera=cam,
                        laser=laser)

        plt.figure()
        plt.imshow(imgA, cmap='gray')
        plt.colorbar()
        for p in particles[particles.mask]:
            plt.scatter(p.x, p.y, s=100, c='g', marker='x')

        for p in particles[~particles.mask]:
            plt.scatter(p.x, p.y, s=100, c='r', marker='x')

        plt.show()

        # n_particles = 5
        # particles = Particles(
        #     x=np.random.uniform(0, cam.nx - 1, n_particles),
        #     y=np.random.uniform(0, cam.ny - 1, n_particles),
        #     z=np.random.uniform(-laser.width, laser.width, n_particles),
        #     size=np.ones(n_particles) * 2
        # )
#
# __this_dir__ = pathlib.Path(__file__).parent
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
#     def tearDown(self) -> None:
#         """delete created files"""
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
