import numpy as np
import xarray as xr


class CorrelationPeak:

    def __init__(self, data):
        self.data = data
        self.j_sub, self.i_sub = get_sub_peak_position(data)


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


def get_sub_peak_position(corr):
    eps = 1e-7
    corr = corr + eps
    # subp_peak_position = (np.nan, np.nan)
    peak1_i = 1
    peak1_j = 1

    c = corr[peak1_i, peak1_j].data
    cl = corr[peak1_i - 1, peak1_j].data
    cr = corr[peak1_i + 1, peak1_j].data
    cd = corr[peak1_i, peak1_j - 1].data
    cu = corr[peak1_i, peak1_j + 1].data

    nom1 = np.log(cl) - np.log(cr)
    den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr)
    nom2 = np.log(cd) - np.log(cu)
    den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu)

    subp_peak_position = (
        peak1_i + np.divide(nom1, den1, out=np.zeros(1),
                            where=(den1 != 0.0))[0],
        peak1_j + np.divide(nom2, den2, out=np.zeros(1),
                            where=(den2 != 0.0))[0],
    )
    return subp_peak_position


class CorrelationPlane:

    def __init__(self, data):
        self.data = xr.DataArray(data,
                                 dims=('iy', 'ix'),
                                 coords={'iy': np.arange(0, data.shape[0]),
                                         'ix': np.arange(0, data.shape[1])})
        self.highest_peak = None

        self.i, self.j = get_integer_peak(self.data)

        peak_arr = self.data[self.j - 1:self.j + 2, self.i - 1: self.i + 2]

        self.highest_peak = CorrelationPeak(
            data=peak_arr
        )
