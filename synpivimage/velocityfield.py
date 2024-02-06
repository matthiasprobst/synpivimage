import numpy as np
import xarray as xr
from scipy import interpolate

from .core import ParticleInfo, SynPivConfig


class VelocityField:

    def __init__(self, x, y, z, u, v, w):
        self.x = x  # 1d!
        self.y = y  # 1d!
        self.z = z  # 1d!
        self.u = xr.DataArray(data=u,
                              dims=('iz', 'iy', 'ix'),
                              coords={'iz': self.z,
                                      'iy': self.y,
                                      'ix': self.x})
        self.v = xr.DataArray(data=v,
                              dims=('iz', 'iy', 'ix'),
                              coords={'iz': self.z,
                                      'iy': self.y,
                                      'ix': self.x})
        self.w = xr.DataArray(data=w,
                              dims=('iz', 'iy', 'ix'),
                              coords={'iz': self.z,
                                      'iy': self.y,
                                      'ix': self.x})
        assert self.u.ndim == 3 == self.v.ndim == 3 == self.w.ndim

    def displace(self, cfg: SynPivConfig, part_info: ParticleInfo):
        """New particles in the 'off' region are created, velocity data
        is interpolated on them"""

        # bounding box is size of velocity field

        def _generate_off_particles(n_particles):
            n_particles = int(n_particles)
            laser_zmin = -cfg.laser_width / 2
            laser_zmax = cfg.laser_width / 2

            box_xp = np.empty(n_particles)
            box_yp = np.empty(n_particles)
            box_zp = np.empty(n_particles)
            i = 0
            while i < n_particles:
                # generate random x, y, z
                x = np.random.uniform(min(self.x), max(self.x))
                y = np.random.uniform(min(self.y), max(self.y))
                z = np.random.uniform(min(self.z), max(self.z))
                if (laser_zmin < z < laser_zmax) and (0 < x < cfg.nx) and (0 < y < cfg.ny):
                    box_xp[i] = x
                    box_yp[i] = y
                    box_zp[i] = z
                    i += 1
            return box_xp, box_yp, box_zp

        dx_interpolator = interpolate.RegularGridInterpolator((self.u.iz,
                                                               self.u.iy,
                                                               self.u.ix),
                                                              self.u.data,
                                                              method='linear')
        # dx_interpolated = dx_interpolator((box_zp, box_yp, box_xp))
        dy_interpolator = interpolate.RegularGridInterpolator((self.v.iz,
                                                               self.v.iy,
                                                               self.v.ix),
                                                              self.v.data,
                                                              method='linear')
        # dy_interpolated = dy_interpolator((box_zp, box_yp, box_xp))
        dz_interpolator = interpolate.RegularGridInterpolator((self.w.iz,
                                                               self.w.iy,
                                                               self.w.ix),
                                                              self.w.data,
                                                              method='linear')
        # dz_interpolated = dz_interpolator((box_zp, box_yp, box_xp))

        # move in-field particles:
        new_orig_part_x = part_info.x + dx_interpolator((part_info.z,
                                                         part_info.y,
                                                         part_info.x))
        new_orig_part_y = part_info.y + dy_interpolator((part_info.z,
                                                         part_info.y,
                                                         part_info.x))
        new_orig_part_z = part_info.z + dz_interpolator((part_info.z,
                                                         part_info.y,
                                                         part_info.x))

        # determine how many particles left the field
        laser_zmin = -cfg.laser_width / 2
        laser_zmax = cfg.laser_width / 2
        laser_area = (new_orig_part_z > laser_zmin) & (new_orig_part_z < laser_zmax) & (new_orig_part_x > 0) & (
                new_orig_part_x < cfg.nx) & (new_orig_part_y >= 0) & (new_orig_part_y < cfg.ny)

        remaining_orig_x = new_orig_part_x[laser_area]
        remaining_orig_y = new_orig_part_y[laser_area]
        remaining_orig_z = new_orig_part_z[laser_area]
        n_out_of_plane = len(part_info.z) - len(remaining_orig_z)
        print(f'Number of particles left the field: {n_out_of_plane}')
        print(f'Percentage of particles left the field: {n_out_of_plane / len(part_info.z) * 100}%')

        target_ppp = cfg.particle_number / (cfg.nx * cfg.ny * cfg.laser_width)
        target_particle_number = cfg.particle_number
        # current ppp:
        current_particle_number = len(remaining_orig_z)
        current_ppp = current_particle_number / (cfg.nx * cfg.ny * cfg.laser_width)

        n_missing = int(target_particle_number - current_particle_number)

        new_particles_x = np.array([])
        new_particles_y = np.array([])
        new_particles_z = np.array([])

        while n_missing > 0:
            # generate particles outside the field:

            # generate new particles
            _n_generate_particles = min(100, n_missing)
            box_xp, box_yp, box_zp = _generate_off_particles(_n_generate_particles)
            while len(box_xp) == 0:
                print('No particles generated, trying again')
                box_xp, box_yp, box_zp = _generate_off_particles(_n_generate_particles)
            xnew = box_xp + dx_interpolator((box_zp, box_yp, box_xp))
            ynew = box_yp + dy_interpolator((box_zp, box_yp, box_xp))
            znew = box_zp + dz_interpolator((box_zp, box_yp, box_xp))

            laser_area = (znew > laser_zmin) & (znew < laser_zmax) & (xnew > 0) & (xnew < cfg.nx) & (
                    ynew >= 0) & (ynew < cfg.ny)

            n_new = np.sum(laser_area)

            n_add = min(n_missing, n_new)

            # keep those inside the laser area:
            new_particles_x = np.concatenate((new_particles_x, xnew[laser_area][0:n_add]))
            new_particles_y = np.concatenate((new_particles_y, ynew[laser_area][0:n_add]))
            new_particles_z = np.concatenate((new_particles_z, znew[laser_area][0:n_add]))

            current_particle_number += n_add
            if n_add > 0:
                print(f'Added {np.sum(laser_area)} particles')

            n_missing = int(target_particle_number - current_particle_number)
        print('Done adding particles')
        print(f'Number of particles added: {len(new_particles_z)}')
        total_count = len(new_particles_z) + len(remaining_orig_z)
        print(f'Total number of particles: {total_count}')
        print(f'ppp: {total_count / (cfg.nx * cfg.ny)} (target={cfg.particle_number / (cfg.nx * cfg.ny)})')

        return ParticleInfo(x=np.concatenate((remaining_orig_x, new_particles_x)),
                            y=np.concatenate((remaining_orig_y, new_particles_y)),
                            z=np.concatenate((remaining_orig_z, new_particles_z)),
                            size=cfg.particle_size_mean * np.ones(total_count))  # TODO: FIXME!


class ConstantField:
    def __init__(self, dx: float, dy: float, dz: float):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def displace(self, cfg: SynPivConfig, part_info: ParticleInfo):
        laser_zmin = -cfg.laser_width / 2
        laser_zmax = cfg.laser_width / 2

        initial_x = part_info.x
        initial_y = part_info.y
        initial_z = part_info.z
        initial_size = part_info.size

        print(f'Laser zmin: {laser_zmin}, zmax: {laser_zmax}')
        print(f'Current zmin: {min(initial_z)}, zmax: {max(initial_z)}')

        particle_number = len(initial_x)
        particle_depth_density = particle_number / (laser_zmax - laser_zmin)  # particles per laser depth

        source_x = initial_x - np.ceil(self.dx)
        source_y = initial_y - np.ceil(self.dy)
        source_z = initial_z - np.ceil(self.dz)

        def _create_new_particles_around_fov(n_particles: int = None):
            n_particles = int(n_particles)
            laser_zmin = -cfg.laser_width / 2
            laser_zmax = cfg.laser_width / 2

            box_xp = np.empty(n_particles)
            box_yp = np.empty(n_particles)
            box_zp = np.empty(n_particles)

            _minx = - np.ceil(self.dx)
            _maxx = cfg.nx + np.ceil(self.dx)
            _miny = - np.ceil(self.dy)
            _maxy = cfg.ny + np.ceil(self.dy)
            _minz = - np.ceil(self.dz)
            _maxz = cfg.laser_width / 2 + np.ceil(self.dz)

            i = 0
            loop_count = 0
            while i < n_particles:
                loop_count += 1
                if loop_count > 10 ** 6:
                    raise ValueError('Too many iterations')
                # generate random x, y, z
                x = np.random.uniform(_minx, max(_maxx, cfg.nx))
                y = np.random.uniform(_miny, max(_maxy, cfg.ny))
                z = np.random.uniform(_minz, max(laser_zmax, _maxz))

                if (laser_zmin < z < laser_zmax) and (0 < x < cfg.nx) and (0 < y < cfg.ny):
                    box_xp[i] = x
                    box_yp[i] = y
                    box_zp[i] = z
                    i += 1
            return box_xp, box_yp, box_zp

            # # fac = 0
            # box_x_min = min(min(source_x), min(initial_x))  # - fac * np.abs(dx)
            # box_x_max = max(max(source_x), max(initial_x))  # + fac * np.abs(dx)
            # box_y_min = min(min(source_y), min(initial_y))  # - fac * np.abs(dy)
            # box_y_max = max(max(source_y), max(initial_y))  # + fac * np.abs(dy)
            # box_z_min = min(min(source_z), min(initial_z))  # - fac * np.abs(dz)
            # box_z_max = max(max(source_z), max(initial_z))  # + fac * np.abs(dz)
            #
            # # print('Bounding box: ', box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max)
            #
            # # create random particles in the box
            # dz_box = box_z_max - box_z_min
            # if n_particles_box is None:
            #     n_particles_box = particle_depth_density * dz_box
            # box_zp = np.random.uniform(box_z_min, box_z_max, int(n_particles_box))
            # box_xp = np.random.uniform(box_x_min, box_x_max, int(n_particles_box))
            # box_yp = np.random.uniform(box_y_min, box_y_max, int(n_particles_box))
            #
            # # remove the particles which are inside the laser area
            # laser_area = (box_zp > laser_zmin) & (box_zp < laser_zmax) & (box_xp > 0) & (box_xp < cfg.nx) & (
            #         box_yp >= 0) & (
            #                      box_yp < cfg.ny)
            #
            # box_xp = box_xp[~laser_area]
            # box_yp = box_yp[~laser_area]
            # box_zp = box_zp[~laser_area]
            # return box_xp, box_yp, box_zp

        # move old particles:
        x_old = initial_x
        y_old = initial_y
        z_old = initial_z
        x_new = x_old + self.dx
        y_new = y_old + self.dy
        z_new = z_old + self.dz
        print('dz', self.dz, 'znew_min', np.min(z_new), 'znew_max', np.max(z_new))
        # what is leaving in z direction?
        leaving_z = (z_new < laser_zmin) | (z_new >= laser_zmax)
        nz_loss = np.sum(leaving_z)
        print(f'out of plane loss: {nz_loss} ({nz_loss / len(z_new) * 100} %)')
        nx_loss = np.sum((x_new < 0) | (x_new >= cfg.nx))
        print(f'out of x plane loss: {nx_loss} ({nx_loss / len(x_new) * 100} %)')
        ny_loss = np.sum((y_new < 0) | (y_new >= cfg.ny))
        print(f'out of y plane loss: {ny_loss} ({ny_loss / len(y_new) * 100} %)')

        # keep those inside the laser sheet
        laser_area = (z_new > laser_zmin) & (z_new < laser_zmax) & (x_new >= 0) & (x_new < cfg.nx) & (y_new >= 0) & (
                y_new < cfg.ny)
        x = x_new[laser_area]
        y = y_new[laser_area]
        z = z_new[laser_area]
        size = initial_size[laser_area]

        # move new particles if they fall inside the new laser sheet until ppp is reached
        missing_particles = cfg.particle_number - len(x)

        i = 0

        def generate_size(cfg):
            # TODO: Fix this
            return cfg.particle_size_mean

        while missing_particles > 0:
            box_xp, box_yp, box_zp = _create_new_particles_around_fov(missing_particles)

            # move off-particles:
            new_part_x = box_xp + self.dx
            new_part_y = box_yp + self.dy
            new_part_z = box_zp + self.dz

            # check if they are within laser sheet:
            laser_area = (new_part_z > laser_zmin) & (new_part_z < laser_zmax) & (new_part_x > 0) & (
                    new_part_x < cfg.nx) & (new_part_y >= 0) & (new_part_y < cfg.ny)
            # new_part_x = new_part_x[laser_area]
            # new_part_y = new_part_y[laser_area]
            # new_part_z = new_part_z[laser_area]

            x = np.concatenate((x, new_part_x[laser_area]))
            y = np.concatenate((y, new_part_y[laser_area]))
            z = np.concatenate((z, new_part_z[laser_area]))
            # TODO: fix generating corret particle sizes!
            size = np.concatenate((size, np.ones(np.sum(laser_area)) * cfg.particle_size_mean))

            missing_particles -= np.sum(laser_area)

        print(f"Number of particles inside the laser sheet: {len(x)}")
        print(f"ppp: {len(x) / (cfg.nx * cfg.ny)}")

        return ParticleInfo(x, y, z, size)

# class RandomUniformField:
#     def __init__(self, dx: float, dy: float, dz: float):
#         self.dx = dx
#         self.dy = dy
#         self.dz = dz
#
#     def displace(self, part_info: ParticleInfo):
#
#         return VelocityField(dx=np.random.uniform(dx[0], dx[1], size=len(part_infoA)),
#                              dy=np.random.uniform(dy[0], dy[1], size=len(part_infoA)),
#                              dz=np.random.uniform(dz[0], dz[1], size=len(part_infoA)))

#
# u = xr.DataArray(data=np.random.rand(10, 10) * 10,
#                  dims=('iy', 'ix'),
#                  coords={'iy': np.arange(0, 10),
#                          'ix': np.arange(0, 10)})
# v = xr.DataArray(data=np.random.rand(10, 10) * 10 + 10,
#                  dims=('iy', 'ix'),
#                  coords={'iy': np.arange(0, 10),
#                          'ix': np.arange(0, 10)})
#
# # particle data:
# xp = np.random.uniform(0, 10, 100)
# yp = np.random.uniform(0, 10, 100)
#
# # interpolate the velocity field to the particle positions
# up = u.interp(ix=xp, iy=yp)
# vp = v.interp(ix=xp, iy=yp)
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.quiver(u.ix, u.iy, u.data, v.data)
# for _xp, _yp in zip(xp, yp):
#     plt.quiver(_xp, _yp, up.sel(ix=_xp, iy=_yp), vp.sel(ix=_xp, iy=_yp), color='r')
# plt.show()
#
# # 1d case (velocity on simulatio boundary):
# x = np.arange(0, 10)
# u = xr.DataArray(data=np.random.random(10) * 10,
#                  dims=('ix',),
#                  coords={'ix': x})
# v = xr.DataArray(data=np.random.random(10) * 10 + 10,
#                  dims=('ix',),
#                  coords={'ix': x})
# xp = np.random.uniform(-2, 12, 10)
#
# up = u.interp(ix=xp)
# vp = v.interp(ix=xp)
#
# plt.figure()
# plt.quiver(u.ix, np.zeros_like(u.ix), u.data, v.data)
# plt.quiver(xp, xp*0, up, vp, color='r')
# plt.show()
