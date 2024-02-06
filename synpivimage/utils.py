import numpy as np

PMIN_ALLOWED: float = 0.1


def generate_particle_size_distribution(mean, std, n):
    pmin = max(PMIN_ALLOWED, mean - 3 * std),
    pmax = mean + 3 * std
    if std > 0:
        particle_sizes = np.random.normal(mean, std, n)
    else:
        particle_sizes = np.ones(n) * mean
    # iout = np.argwhere((particle_sizes < pmin) | (particle_sizes > pmax))
    # for i in iout[:, 0]:
    #     dp = np.random.normal(mean, std)
    #     while dp < pmin or dp > pmax:
    #         dp = np.random.normal(mean, std)
    #     particle_sizes[i] = dp

    particle_sizes[particle_sizes < pmin] = pmin
    particle_sizes[particle_sizes > pmax] = pmax
    return particle_sizes
