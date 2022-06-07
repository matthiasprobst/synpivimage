import numpy as np

from synpivimage import DEFAULT_CFG
from synpivimage import build_ConfigManager


def test_build_config_manager():
    cfg = DEFAULT_CFG
    cfg['nx'] = 16
    cfg['ny'] = 16
    particle_number_range = ('particle_number', np.linspace(1, cfg['ny'] * cfg['nx'], 101).astype(int))
    CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=1)
    assert len(CFG) == 101
    CFG = build_ConfigManager(cfg, [particle_number_range, ], per_combination=2)
    assert len(CFG) == 101*2

    generated_particle_number = [cfg['particle_number'] for cfg in CFG.cfgs]
    assert np.array_equal(np.unique(np.sort(generated_particle_number)), particle_number_range[1])
