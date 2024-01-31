import pathlib
import unittest

import h5rdmtoolbox as h5tbx
import matplotlib.pyplot as plt
# import h5py
import numpy as np

from synpivimage import DEFAULT_CFG
from synpivimage import build_ConfigManager


# import warnings


# from synpivimage.conventions import Layout


class TestCore(unittest.TestCase):

    def setUp(self) -> None:
        """setup"""
        hdf_filename = pathlib.Path('ds_000000.hdf')
        if hdf_filename.exists():
            hdf_filename.unlink(missing_ok=True)

    def tearDown(self) -> None:
        """delete created files"""
        hdf_filename = pathlib.Path('ds_000000.hdf')
        if hdf_filename.exists():
            hdf_filename.unlink(missing_ok=True)

    def test_build_config_manager(self):
        cfg = DEFAULT_CFG
        cfg['nx'] = 16
        cfg['ny'] = 16
        particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 101).astype(int))
        CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
        assert len(CFG) == 101
        CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=2)
        assert len(CFG) == 101 * 2

        generated_particle_number = [cfg['particle_number'] for cfg in CFG.cfgs]
        assert np.array_equal(np.unique(np.sort(generated_particle_number)), particle_number_range[1])

    def test_generate(self):
        cfg = DEFAULT_CFG
        cfg['nx'] = 16
        cfg['ny'] = 16
        particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 5).astype(int))
        CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
        CFG.generate('.', nproc=1)

        hdf_filename = 'ds_000000.hdf'

        try:
            from h5rdmtoolbox.conventions.cflike import StandardNameTable
            snt = StandardNameTable.from_gitlab(url='https://git.scc.kit.edu',
                                                file_path='particle_image_velocimetry-v1.yaml',
                                                project_id='35942',
                                                ref_name='main')

            snt.check_file(hdf_filename)
        except ImportError:
            warnings.warn('Standard names could not be checked as h5rdmtoolbox is not installed')

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

        h5l = Layout()
        h5l.check_file(hdf_filename)
        self.assertEqual(h5l.n_issues, 0)
        h5l.report()

    def test_generate_second_image(self):
        cfg = DEFAULT_CFG
        cfg['nx'] = 16
        cfg['ny'] = 16
        cfg['laser_width'] = 2
        particle_numbers = np.linspace(1, cfg['ny'] * cfg['nx'], 5).astype(int)
        CFG = build_ConfigManager(initial_cfg=cfg,
                                  variation_dict={'particle_number': particle_numbers},
                                  per_combination=1)

        # with h5tbx.File(hdf_filenames[0]) as h5:
        #     get x,y,z,size from hdf file and feed to to image B generation

        hdf_filenames = CFG.generate(
            data_directory='.',
            nproc=1,
            particle_info={'x': [8, 9, 10, 11, 12],
                           'y': [8, 8, 8, 8, 8],
                           'z': [0, -0.5, -1, -1.5, -2],
                           'size': [2.5, 2.5, 2.5, 2.5, 2.5]})
        with h5tbx.File(hdf_filenames[0]) as h5:
            h5.images[0, ...].plot()
        plt.show()
