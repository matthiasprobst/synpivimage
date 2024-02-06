import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import rfft2, irfft2, fftshift

import synpivimage as spi
from lib.corr import CorrelationPlane
from lib.plotting import gauss3ptfit
from src.main import Ui_MainWindow
from synpivimage.core import particle_intensity
from synpivimage.velocityfield import ConstantField

__this_dir__ = pathlib.Path(__file__).parent
INIT_DIR = __this_dir__


def generate_correlation(imgA, imgB):
    f2a = np.conj(rfft2(imgA))
    f2b = rfft2(imgB)
    return fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))


class Ui(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, root_dir):
        self.curr_img_index = 0

        super(Ui, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Synthetic PIV Image Generator')

        self.nx.setMinimum(4)
        self.nx.setValue(16)
        self.ny.setMinimum(4)
        self.ny.setValue(16)
        self.nx.setMaximum(1000)
        self.ny.setMaximum(1000)
        self.particle_number.setMinimum(1)
        self.particle_number.setValue(10)
        self.particle_number.setMaximum(10 ** 5)
        self.particle_size_mean.setMinimum(0.5)
        self.particle_size_mean.setValue(2.5)
        self.particle_size_std.setMinimum(0.0)
        self.particle_size_std.setValue(0.0)
        self.laser_width.setMinimum(0)
        self.laser_width.setValue(2)
        self.laser_shape_factor.setMinimum(1)
        self.laser_shape_factor.setMaximum(10 ** 6)
        self.laser_shape_factor.setValue(10000)
        self.laser_shape_factor.setToolTip('gaussian: 2, top hat: 10000')

        self.particle_count.setMinimum(1)
        self.particle_count.setMaximum(2 ** 16)
        self.particle_count.setValue(1000)

        self.dx.setMinimum(-100)
        self.dx.setMaximum(100)
        self.dx.setValue(0.3)

        self.dy.setMinimum(-100)
        self.dy.setMaximum(100)
        self.dy.setValue(0.6)

        self.dz.setMinimum(-100)
        self.dz.setMaximum(100)
        self.dz.setValue(0.0)

        self.n_imgs.setMinimum(1)
        self.n_imgs.setValue(1)
        self.n_imgs.setMaximum(10 ** 6)

        self.baseline.setMinimum(0)
        self.baseline.setMaximum(2 ** 16)
        self.baseline.setValue(50)

        self.darknoise.setMinimum(0)
        self.darknoise.setMaximum(2 ** 16 / 2)
        self.darknoise.setValue(10)

        self.bit_depth.setMinimum(4)
        self.bit_depth.setValue(16)
        self.bit_depth.setMaximum(32)
        n_imgs = self.n_imgs.value()
        nx = self.nx.value()
        ny = self.ny.value()
        self.imgsA = np.empty(shape=(n_imgs, ny, nx))
        self.imgsB = np.empty(shape=(n_imgs, ny, nx))
        self.correlations = np.empty(shape=(n_imgs, ny, nx))
        self.partA_infos = [{}] * n_imgs
        self.partB_infos = [{}] * n_imgs

        plotting_layout1 = QHBoxLayout(self.plotwidget1)
        plotting_layout2 = QHBoxLayout(self.plotwidget2)

        dummy_arr = np.random.random((self.nx.value(), self.ny.value()))

        self.figures = []
        self.axes = []
        self.canvas = []
        self.ims = []
        self.cax = []
        for i in range(3):
            figure, ax = plt.subplots(tight_layout=True)
            self.figures.append(figure)
            self.axes.append(ax)
            self.canvas.append(FigureCanvas(figure))
            plotting_layout1.addWidget(self.canvas[-1])

            im = ax.imshow(dummy_arr, cmap='gray')
            divider = make_axes_locatable(ax)
            self.cax.append(divider.append_axes("right", size="5%", pad=0.05))
            _ = plt.colorbar(im, cax=self.cax[-1])

            self.ims.append(im)

        figure, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
        self.figures.append(figure)
        self.axes.append(ax)
        self.canvas.append(FigureCanvas(figure))
        plotting_layout2.addWidget(self.canvas[-1])
        ax[0].plot([1, 2, 3])
        ax[1].plot([1, 2, 3])

        for i in range(2):
            figure, ax = plt.subplots(tight_layout=True)
            self.figures.append(figure)
            self.axes.append(ax)
            self.canvas.append(FigureCanvas(figure))
            plotting_layout2.addWidget(self.canvas[-1])
            ax.plot([1, 2, 3])

        self.update()

        self._root_dir = root_dir

        self.btn_update.clicked.connect(self.update)
        # self.particle_number.valueChanged.connect(self.update)
        # self.particle_number.editingFinished.connect(self.update)

        self.show()

    def keyPressEvent(self, event):
        # This function will be called when any key is pressed
        if event.key() == Qt.Key_F5:
            # Handle F5 key press
            self.update()
        if event.key() == Qt.Key_F6:
            # Handle F6 key press
            if self.curr_img_index == self.imgsA.shape[0]:
                self.curr_img_index = 0
            else:
                self.curr_img_index += 1
            self.update_plot()

    def get_config(self):
        return spi.config.SynPivConfig(
            ny=self.ny.value(),
            nx=self.nx.value(),
            bit_depth=self.bit_depth.value(),
            dark_noise=self.darknoise.value(),
            image_particle_peak_count=self.particle_count.value(),
            laser_shape_factor=self.laser_shape_factor.value(),
            laser_width=self.laser_width.value(),
            noise_baseline=self.baseline.value(),
            particle_number=self.particle_number.value(),
            particle_position_file=None,
            particle_size_illumination_dependency=True,
            particle_size_mean=self.particle_size_mean.value(),
            particle_size_std=self.particle_size_std.value(),
            qe=1.,
            sensitivity=1.,
            shot_noise=self.shotnoise.isChecked())

    def generate_images(self):
        cfg = self.get_config()
        assert cfg.nx == self.nx.value()
        assert cfg.ny == self.ny.value()
        n_imgs = self.n_imgs.value()
        imgs_shape = (n_imgs, cfg.ny, cfg.nx)
        if self.imgsA.shape != imgs_shape:
            print('reallocate image arrays')
            self.imgsA = np.empty(shape=imgs_shape)
            self.imgsB = np.empty(shape=imgs_shape)
            self.correlations = np.empty(shape=imgs_shape)
            self.partA_infos = [{}] * n_imgs
            self.partB_infos = [{}] * n_imgs

        for i in range(n_imgs):
            imgA, _, partA = spi.generate_image(
                cfg
            )

            cfield = ConstantField(dx=self.dx.value(), dy=self.dy.value(), dz=self.dz.value())
            imgB, _, partB = spi.generate_image(
                cfg,
                particle_data=cfield.displace(cfg, partA)
            )
            self.imgsA[i, ...] = imgA
            self.imgsB[i, ...] = imgB

            self.partA_infos[i] = partA
            self.partB_infos[i] = partB

    def update(self):
        self.curr_img_index = 0
        # generate images
        self.generate_images()
        # compute correlations:
        for i in range(self.imgsA.shape[0]):
            self.correlations[i, ...] = generate_correlation(
                self.imgsA[i, ...],
                self.imgsB[i, ...]
            )
        # plot images
        self.update_plot()

    def update_plot(self):
        self._plot_imgA()
        self._plot_imgB()
        self._plot_correlation()

    def _plot_imgA(self):
        # plot imgA to self.plot11 widget
        im = self.axes[0].imshow(self.imgsA[self.curr_img_index], cmap='gray')
        plt.colorbar(im, cax=self.cax[0])
        self.canvas[0].draw()

    def _plot_imgB(self):
        # plot_img(self.imgB, self.axes[1])
        im = self.axes[1].imshow(self.imgsB[self.curr_img_index], cmap='gray')
        plt.colorbar(im, cax=self.cax[1])
        self.canvas[1].draw()

    # @property
    # def correlation(self):
    #     if self.curr_img_index >= self.correlations.shape[0]:
    #         self.curr_img_index = self.correlations.shape[0] - 1
    #     return self.correlations[self.curr_img_index]
    #
    # @property
    # def imgA(self):
    #     if self.curr_img_index >= self.imgsA.shape[0]:
    #         self.curr_img_index = self.imgsA.shape[0] - 1
    #     return self.imgsA[self.curr_img_index]
    #
    # @property
    # def imgB(self):
    #     if self.curr_img_index >= self.imgsB.shape[0]:
    #         self.curr_img_index = self.imgsB.shape[0] - 1
    #     return self.imgsB[self.curr_img_index]

    def _plot_correlation(self):
        self.axes[5].cla()
        self.axes[2].cla()
        im = self.axes[2].imshow(self.correlations[self.curr_img_index], cmap='gray')
        plt.colorbar(im, cax=self.cax[2])
        self.canvas[2].draw()
        corr = CorrelationPlane(self.correlations[self.curr_img_index])
        corr.data[corr.j, :].plot(ax=self.axes[5], color='r')
        corr.data[:, corr.i].plot(ax=self.axes[5], color='b')

        corr.highest_peak.data[1, :].plot.scatter(ax=self.axes[5], color='r')
        self.axes[5].scatter(corr.i, np.max(corr.highest_peak.data), color='r', marker='^')

        corr.highest_peak.data[:, 1].plot.scatter(ax=self.axes[5], color='b')
        self.axes[5].scatter(corr.j, np.max(corr.highest_peak.data), color='b', marker='^')

        self.axes[2].scatter(corr.i,
                             corr.j,
                             marker='+',
                             color='r')

        nx = self.nx.value()
        ny = self.ny.value()

        try:
            g1 = gauss3ptfit(corr.highest_peak.data[1, :])
            _x = np.linspace(0, nx, 100)
            self.axes[5].plot(_x, g1(_x - corr.i), color='r', linestyle='--')
        except RuntimeError as e:
            print(e)
        try:
            g2 = gauss3ptfit(corr.highest_peak.data[:, 1])

            _y = np.linspace(0, ny, 100)
            self.axes[5].plot(_y, g2(_y - corr.j), color='b', linestyle='--')
        except RuntimeError as e:
            print(e)
        _min = min(corr.i, corr.j) - 5
        _max = max(corr.i, corr.j) + 5
        self.axes[5].set_xlim(_min, _max)
        self.canvas[5].draw()

        # # canvas 3: plot scatter of displacements:
        for ax in self.axes[3]:
            ax.cla()

        self.axes[4].cla()

        estimated_dx = np.empty(shape=(self.imgsA.shape[0]))
        estimated_dy = np.empty(shape=(self.imgsA.shape[0]))
        for i in range(self.imgsA.shape[0]):
            corr = CorrelationPlane(self.correlations[i, ...])
            # self.axes[4].scatter(corr.j + corr.highest_peak.j_sub, corr.i + corr.highest_peak.i_sub, color='k',
            #                      marker='+')

            dx = corr.i + corr.highest_peak.i_sub
            dy = corr.j + corr.highest_peak.j_sub
            estimated_dx[i] = dx
            estimated_dy[i] = dy

            # sub_dx = dx - round(dx)
            # sub_dy = dy - round(dy)

        sub_pixel_dx = estimated_dx - nx / 2 - 1
        sub_pixel_dy = estimated_dy - ny / 2 - 1
        self.axes[4].scatter(sub_pixel_dx, sub_pixel_dy, color='r', marker='o', alpha=0.5)

        true_dx, true_dy = self.dx.value(), self.dy.value()
        self.axes[4].scatter(true_dx, true_dy, color='k', marker='+')
        self.axes[4].set_xlim(true_dx - 1, true_dx + 1)
        self.axes[4].set_ylim(true_dy - 1, true_dy + 1)

        # compute RMS values
        rms_x = np.sqrt(np.sum((sub_pixel_dx - true_dx) ** 2) / (max(1, len(sub_pixel_dx) - 1)))
        rms_y = np.sqrt(np.sum((sub_pixel_dy - true_dy) ** 2) / (max(1, len(sub_pixel_dy) - 1)))
        rms = np.sqrt(rms_x ** 2 + rms_y ** 2)
        self.axes[4].text(true_dx + 0.5, true_dy + 0.5, f'{rms:.3f}')

        #
        # xlims = self.axes[4].get_xlim()
        # ylims = self.axes[4].get_ylim()
        # self.axes[4].hlines(xlims[0], xlims[1], self.dx.value() - self.nx.value() / 2, color='k', linestyle='--')
        # self.axes[4].vlines(self.dy.value() - self.ny.value() / 2, ylims[0], ylims[1], color='k', linestyle='--')
        # self.canvas[3].draw()
        # self.canvas[4].draw()
        z = np.linspace(-2 * self.laser_width.value(), 2 * self.laser_width.value(), 1000)
        laser_intensity = particle_intensity(z=z,
                                             beam_width=self.laser_width.value(),  # laser beam width
                                             s=self.laser_shape_factor.value(),  # shape factor
                                             dp=None)
        self.axes[3][0].plot(z, laser_intensity)
        # self.axes[4][0].plot(z, laser_intensity)

        self.axes[3][1].scatter(self.partA_infos[self.curr_img_index].z, self.partA_infos[self.curr_img_index].y,
                                color='b', alpha=0.5, s=10, marker='o')
        # draw vlines for laser
        self.axes[3][1].vlines(-self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')
        self.axes[3][1].vlines(self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')

        self.axes[3][1].scatter(self.partB_infos[self.curr_img_index].z, self.partB_infos[self.curr_img_index].y,
                                color='r', alpha=0.5, s=10, marker='o')
        self.axes[3][1].vlines(-self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')
        self.axes[3][1].vlines(self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')

        self.canvas[2].draw()
        self.canvas[3].draw()
        self.canvas[4].draw()


def start(*args, wd=None, console: bool = True):
    """call the gui"""
    print('Preparing piv2hdf gui ...')
    if wd is None:
        root_dir = pathlib.Path.cwd()
    else:
        root_dir = pathlib.Path(wd)

    if not root_dir.exists():
        print(f'cannot start gui on that path: {root_dir}')

    if console:
        print('Initializing gui from console...')
        app = QtWidgets.QApplication([*args, ])
    else:
        if len(args[0]) == 2:
            root_dir = pathlib.Path(args[0][0]).parent
            root_dir = root_dir.joinpath(args[0][1])
        else:
            root_dir = INIT_DIR
        print('Initializing gui from python script...')
        app = QtWidgets.QApplication(*args, )

    _ = Ui(root_dir)

    print('Starting gui ...')
    app.exec_()
    print('Closed gui ...')


if __name__ == "__main__":
    start(sys.argv)
