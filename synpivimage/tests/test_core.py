import pathlib
import unittest

import h5py
import numpy as np

from synpivimage import DEFAULT_CFG
from synpivimage import __version__
from synpivimage import build_ConfigManager


class TestCore(unittest.TestCase):

    def test_version(self):
        try:
            from importlib.metadata import version as _version
            assert __version__ == '0.1.9'
        except ImportError:
            assert __version__ == '999'

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
        if pathlib.Path('ds_000000.hdf').exists():
            pathlib.Path('ds_000000.hdf').unlink()
        # python3.8 only:
        # pathlib.Path('ds_000000.hdf').unlink(missing_ok=True)
        cfg = DEFAULT_CFG
        cfg['nx'] = 16
        cfg['ny'] = 16
        particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 5).astype(int))
        CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
        CFG.to_hdf('.', nproc=1)
        with h5py.File('ds_000000.hdf') as h5:
            self.assertIn('images', h5)
            self.assertIn('image_index', h5)
            self.assertTrue(h5['images'].attrs['standard_name'], 'synthetic_particle_image')
            for dsname in h5.keys():
                if isinstance(h5[dsname], h5py.Dataset) and dsname == 'images':
                    assert h5[dsname].dims[0][0] == h5['image_index']
                    assert h5[dsname].dims[0][1] == h5['nparticles']
            assert h5['images'].dims[1][0] == h5['iy']
            assert h5['images'].dims[2][0] == h5['ix']
        pathlib.Path('ds_000000.hdf').unlink(missing_ok=True)
