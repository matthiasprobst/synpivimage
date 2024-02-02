import h5py
import h5rdmtoolbox as h5tbx
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import unittest

import synpivimage.core
from synpivimage import DEFAULT_CFG
from synpivimage import build_ConfigManager, generate_image

__this_dir__ = pathlib.Path(__file__).parent


class TestCore(unittest.TestCase):

    def setUp(self) -> None:
        """setup"""
        hdf_filenames = __this_dir__.glob('ds*.hdf')
        for hdf_filename in hdf_filenames:
            hdf_filename.unlink(missing_ok=True)

    def tearDown(self) -> None:
        """delete created files"""
        hdf_filenames = __this_dir__.glob('ds*.hdf')
        for hdf_filename in hdf_filenames:
            hdf_filename.unlink(missing_ok=True)

    def test_write_read_config(self):
        from synpivimage.core import generate_default_yaml_file, read_config, SynPivConfig
        filename = generate_default_yaml_file()

        cfg = read_config(filename)
        self.assertIsInstance(cfg, SynPivConfig)

        filename.unlink(missing_ok=True)

    def test_build_config_manager(self):
        cfg = DEFAULT_CFG
        cfg.nx = 16
        cfg.ny = 16
        particle_number_variation = dict(particle_number=np.linspace(1, cfg.ny * cfg.nx, 101).astype(int))
        CFG = build_ConfigManager(initial_cfg=cfg,
                                  variation_dict=particle_number_variation,
                                  per_combination=1)
        assert len(CFG) == 101
        CFG = build_ConfigManager(initial_cfg=cfg,
                                  variation_dict=particle_number_variation,
                                  per_combination=2)
        assert len(CFG) == 101 * 2

        generated_particle_number = [cfg['particle_number'] for cfg in CFG.cfgs]
        assert np.array_equal(np.unique(np.sort(generated_particle_number)),
                              particle_number_variation['particle_number'])

    def test_generate(self):
        cfg = DEFAULT_CFG
        cfg.nx = 16
        cfg.ny = 16
        particle_number_range = {'particle_number': np.linspace(1, cfg.ny * cfg.nx, 5).astype(int)}
        CFG = build_ConfigManager(initial_cfg=cfg,
                                  variation_dict=particle_number_range,
                                  per_combination=1)
        CFG.generate('.', nproc=1)

        hdf_filename = 'ds_000000.hdf'

        with h5py.File(hdf_filename) as h5:
            self.assertIn('images', h5)
            self.assertIn('labels', h5)
            self.assertEqual(h5['labels'].shape, h5['images'].shape)
            np.testing.assert_almost_equal(h5['nparticles'][:], (h5['labels'][...].sum(axis=(1, 2)) / 100).astype(int))
            self.assertIn('image_index', h5)
            self.assertTrue(h5['images'].attrs['standard_name'], 'synthetic_particle_image')
            for ds_name in h5.keys():
                if isinstance(h5[ds_name], h5py.Dataset) and ds_name == 'images':
                    assert h5[ds_name].dims[0][0] == h5['image_index']
                    assert h5[ds_name].dims[0][1] == h5['nparticles']
            assert h5['images'].dims[1][0] == h5['iy']
            assert h5['images'].dims[2][0] == h5['ix']

    def test_generate_second_image(self):
        cfg = DEFAULT_CFG
        cfg.nx = 16
        cfg.ny = 16
        cfg.laser_width = 2
        particle_numbers = np.linspace(1, cfg.ny * cfg.nx * 0.1, 5).astype(int)

        CFG = build_ConfigManager(
            initial_cfg=cfg,
            variation_dict={'particle_number': particle_numbers},
            per_combination=1
        )

        # with h5tbx.File(hdf_filenames[0]) as h5:
        #     get x,y,z,size from hdf file and feed to to image B generation

        hdf_filenamesA = CFG.generate(
            data_directory='.',
            nproc=1,
            suffix='A.hdf',
            particle_info=synpivimage.core.ParticleInfo(
                x=np.array([8, 9, 10, 11, 12]),
                y=np.array([8, 8, 8, 8, 8]),
                z=np.array([0, -0.5, -1, -1.5, -2]),
                size=np.array([2.5, 2.5, 2.5, 2.5, 2.5])
            )
        )
        part_info = synpivimage.core.ParticleInfo.from_hdf(hdf_filenamesA[0])
        [p.displace(dy=2) for p in part_info]
        hdf_filenamesB = CFG.generate(
            data_directory='.',
            suffix='B.hdf',
            nproc=1,
            particle_info=part_info)

        with h5tbx.File(hdf_filenamesA[0]) as h5:
            h5.images[0, ...].plot()
        plt.show()
        with h5tbx.File(hdf_filenamesB[0]) as h5:
            h5.images[0, ...].plot()
        plt.show()

    def test_create_single_image(self):
        cfg = DEFAULT_CFG
        cfg.nx = 16
        cfg.ny = 16
        cfg.laser_width = 2
        cfg.particle_number = 1
        cfg.qe = 1
        cfg.dark_noise = 0
        cfg.noise_baseline = 0
        cfg.shot_noise = False
        cfg.sensitivity = 1
        cfg.image_particle_peak_count = 1000

        imgA, attrsA, part_infoA = generate_image(
            cfg,
            particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
        )
        self.assertEqual(imgA.max(), cfg.image_particle_peak_count)

        cfg.qe = 0.25

        imgA, attrsA, part_infoA = generate_image(
            cfg,
            particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
        )
        self.assertEqual(imgA.max(), cfg.image_particle_peak_count)

        cfg.qe = 1
        cfg.sensitivity = 1 / 4

        imgA, attrsA, part_infoA = generate_image(
            cfg,
            particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
        )
        self.assertEqual(imgA.max(), cfg.image_particle_peak_count)

        # max count = 1000
        cfg.qe = 1
        cfg.sensitivity = 1
        cfg.image_particle_peak_count = 1000

        imgA, attrsA, part_infoA = generate_image(
            cfg,
            particle_data=synpivimage.ParticleInfo(x=8, y=8, z=0, size=2.5)
        )
        self.assertEqual(imgA.max(), cfg.image_particle_peak_count)
        self.assertEqual(imgA[8, 8], cfg.image_particle_peak_count)

        fig, axs = plt.subplots(1, 1)
        imgAmax = imgA.max()
        im = axs.imshow(imgA, cmap='gray', vmin=0, vmax=imgAmax)
        plt.colorbar(im)
        plt.show()

    # def test_create_single_image(self):
    #     cfg = DEFAULT_CFG
    #     cfg.nx = 16
    #     cfg.ny = 16
    #     cfg.laser_width = 2
    #     cfg.particle_number = 5
    #     cfg.qe = 0.25
    #     cfg.dark_noise = 0
    #     cfg.noise_baseline = 100
    #     cfg.relative_laser_intensity = 1000/(2**cfg.bit_depth)
    #     imgA, attrsA, part_infoA = generate_image(
    #         cfg,
    #         particle_data=None
    #     )
    #     part_infoA.displace(dx=2, dy=1, dz=-1)
    #     imgB, attrsB, part_infoB = generate_image(
    #         cfg,
    #         particle_data=part_infoA
    #     )
    #     fig, axs = plt.subplots(1, 2)
    #     imgAmax = imgA.max()
    #     axs[0].imshow(imgA, cmap='gray', vmin=0, vmax=imgAmax)
    #     im = axs[1].imshow(imgB, cmap='gray', vmin=0, vmax=imgAmax)
    #     plt.colorbar(im)
    #     plt.show()
