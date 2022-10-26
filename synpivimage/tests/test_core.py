import pathlib
import unittest
import warnings

import h5py
import numpy as np

from synpivimage import DEFAULT_CFG
from synpivimage import build_ConfigManager
from synpivimage.conventions import Layout


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

    def test_to_hdf(self):
        cfg = DEFAULT_CFG
        cfg['nx'] = 16
        cfg['ny'] = 16
        particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 5).astype(int))
        CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
        CFG.to_hdf('.', nproc=1)

        hdf_filename = 'ds_000000.hdf'

        try:
            import h5rdmtoolbox as h5tbx
            snt = h5tbx.conventions.cflike.standard_name.StandardNameTable.from_gitlab(url='https://git.scc.kit.edu',
                                                                                       file_path='particle_image_velocimetry-v1.yaml',
                                                                                       project_id='35942',
                                                                                       ref_name='main')

            snt.check_file(hdf_filename)
        except ImportError:
            warnings.warn('Standard names could not be checked as h5rdmtoolbox is not installed')

        with h5py.File(hdf_filename) as h5:
            self.assertIn('images', h5)
            self.assertIn('image_index', h5)
            self.assertTrue(h5['images'].attrs['standard_name'], 'synthetic_particle_image')
            for dsname in h5.keys():
                if isinstance(h5[dsname], h5py.Dataset) and dsname == 'images':
                    assert h5[dsname].dims[0][0] == h5['image_index']
                    assert h5[dsname].dims[0][1] == h5['nparticles']
            assert h5['images'].dims[1][0] == h5['iy']
            assert h5['images'].dims[2][0] == h5['ix']

        h5l = Layout()
        h5l.check_file(hdf_filename)
        self.assertEqual(h5l.n_issues, 0)
        h5l.report()
