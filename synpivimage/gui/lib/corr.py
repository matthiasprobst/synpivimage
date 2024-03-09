import abc

import numpy as np
import xarray as xr
from typing import Tuple

EPS = 1e-7


def get_integer_peak(corr):
    corr = np.asarray(corr)
    ind = corr.ravel().argmax(-1)
    peaks = np.array(np.unravel_index(ind, corr.shape[-2:]))

    peaks = np.vstack((peaks[0], peaks[1])).T
    index_list = [(i, v[0], v[1]) for i, v in enumerate(peaks)]
    # peaks_max = np.nanmax(corr, axis=(-2, -1))

    # np.array(index_list), np.array(peaks_max)
    iy, ix = index_list[0][2], index_list[0][1]
    return iy, ix


class PeakFit(abc.ABC):
    """Abstract base class for peak position estimation methods."""

    def __init__(self, j, i):
        self.j = j
        self.i = i

    def get_arr(self, arr):
        arr = np.asarray(arr)
        cl = arr[1, 0]
        cc = arr[1, 1]
        cr = arr[1, 2]
        cu = arr[2, 1]
        cd = arr[0, 1]
        return cl, cc, cr, cu, cd


class Centroid(PeakFit):
    """peak centroid peak position estimation"""

    def __call__(self, arr) -> Tuple[float, float]:
        """assuming highest peak is at the center of the array"""
        cl, cc, cr, cu, cd = self.get_arr(arr)
        nom1 = (self.i - 1) * cl + self.i * cc + (self.i + 1) * cr
        den1 = cl + cc + cr
        nom2 = (self.j - 1) * cd + self.j * cc + (self.j + 1) * cu
        den2 = cu + cc + cd
        subp_peak_position = (
            self.i + np.divide(nom2, den2, out=np.zeros(1),
                      where=(den2 != 0.0))[0],
            self.j + np.divide(nom1, den1, out=np.zeros(1),
                      where=(den1 != 0.0))[0]
        )
        return subp_peak_position


class ParabolicFit(PeakFit):
    """peak position estimation using parabolic fit"""

    def __call__(self, arr) -> Tuple[float, float]:
        """assuming highest peak is at the center of the array"""
        cl, cc, cr, cu, cd = self.get_arr(arr)
        nom1 = cl - cr
        den1 = 2 * cl - 4 * cc + 2 * cr
        nom2 = cd - cu
        den2 = 2 * cu - 4 * cc + 2 * cd
        subp_peak_position = (
            self.j + np.divide(nom2, den2, out=np.zeros(1),
                               where=(den2 != 0.0))[0],
            self.i + np.divide(nom1, den1, out=np.zeros(1),
                               where=(den1 != 0.0))[0]
        )
        return subp_peak_position


class GaussianFit(PeakFit):
    """peak position estimation using Gaussian fit"""

    def __call__(self, arr) -> Tuple[float, float]:
        """assuming highest peak is at the center of the array"""
        cl, cc, cr, cu, cd = self.get_arr(arr)

        nom1 = np.log(cl) - np.log(cr)
        den1 = 2 * np.log(cl) - 4 * np.log(cc) + 2 * np.log(cr)
        nom2 = np.log(cd) - np.log(cu)
        den2 = 2 * np.log(cd) - 4 * np.log(cc) + 2 * np.log(cu)
        subp_peak_position = (
            self.j + np.divide(nom2, den2, out=np.zeros(1),
                               where=(den2 != 0.0))[0],
            self.i + np.divide(nom1, den1, out=np.zeros(1),
                               where=(den1 != 0.0))[0]
        )
        return subp_peak_position


fitting_lib = {'centroid': Centroid,
               'parabolic': ParabolicFit,
               'gaussian': GaussianFit}


class CorrelationPlane:

    def __init__(self, data, fit: str, normalize=True):
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())
        self.data = xr.DataArray(data,
                                 dims=('iy', 'ix'),
                                 coords={'iy': np.arange(0, data.shape[0]),
                                         'ix': np.arange(0, data.shape[1])})
        self.highest_peak = None

        self.i, self.j = get_integer_peak(self.data)

        peak_arr = self.data[self.j - 1:self.j + 2, self.i - 1: self.i + 2]
        if not peak_arr.shape == (3, 3):
            print('fixing peak array')
            if self.j == 0:
                self.j = 1
            elif self.j == data.shape[0] - 1:
                self.j = data.shape[0] - 2
            if self.i == 0:
                self.i = 1
            elif self.i == data.shape[1] - 1:
                self.i = data.shape[1] - 2
            peak_arr = self.data[self.j - 1:self.j + 2, self.i - 1: self.i + 2]

        self.fitter = fitting_lib[fit](self.j, self.i)

        self.j_sub, self.i_sub = self.fitter(peak_arr + EPS)
