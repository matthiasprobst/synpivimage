import pathlib

import h5py
import numpy as np

from synpivimage import DEFAULT_CFG
from synpivimage import __version__
from synpivimage import build_ConfigManager


def test_version():
    try:
        from importlib.metadata import version as _version
        assert __version__ == '0.1.6'
    except ImportError:
        assert __version__ == '999'


def test_build_config_manager():
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


def test_to_hdf():
    if pathlib.Path('ds_000000.hdf').exists():
        pathlib.Path('ds_000000.hdf').unlink()
    # python3.8 only:
    # pathlib.Path('ds_000000.hdf').unlink(missing_ok=True)
    cfg = DEFAULT_CFG
    cfg['nx'] = 16
    cfg['ny'] = 16
    particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 5).astype(int))
    CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
    CFG.to_hdf('.')
    with h5py.File('ds_000000.hdf') as h5:
        assert 'image' in h5
        assert 'image_index' in h5
        for dsname in h5.keys():
            if isinstance(h5[dsname], h5py.Dataset) and dsname not in ('iy', 'ix', 'image_index'):
                print(dsname)
                assert h5[dsname].dims[0][0] == h5['image_index']
        assert h5['image'].dims[1][0] == h5['iy']
        assert h5['image'].dims[2][0] == h5['ix']
    if pathlib.Path('ds_000000.hdf').exists():
        pathlib.Path('ds_000000.hdf').unlink()
    # python3.8 only:
    # pathlib.Path('ds_000000.hdf').unlink(missing_ok=True)
